from typing import Any, Dict, Tuple

import flax.struct as struct
import jax
import jax.numpy as jnp
import mujoco
from gymnax.environments import environment, spaces
from mujoco import mjx

# Load MuJoCo model
_MJ_MODEL = mujoco.MjModel.from_xml_path("scene.xml")
_MJX_MODEL = mjx.put_model(_MJ_MODEL)

# Address constants
ROBOT_QPOS_ADRS = jnp.array(
    [
        _MJ_MODEL.joint("x_pos").qposadr[0],
        _MJ_MODEL.joint("y_pos").qposadr[0],
        _MJ_MODEL.joint("orientation").qposadr[0],
    ]
)
ROBOT_QVEL_ADRS = jnp.array(
    [
        _MJ_MODEL.joint("x_pos").dofadr[0],
        _MJ_MODEL.joint("y_pos").dofadr[0],
        _MJ_MODEL.joint("orientation").dofadr[0],
    ]
)
ROBOT_CTRL_ADRS = jnp.array(
    [
        _MJ_MODEL.actuator("forward_motor").id,
        _MJ_MODEL.actuator("left_motor").id,
        _MJ_MODEL.actuator("orientation_motor").id,
    ]
)
BALL_QPOS_ADRS = jnp.arange(3) + _MJ_MODEL.joint("free_ball").qposadr[0]
BALL_QVEL_ADRS = jnp.arange(6) + _MJ_MODEL.joint("free_ball").dofadr[0]
MASS = _MJ_MODEL.body("robot_base").mass
TARGET_POS = jnp.array([3.0, 2.0])


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_accel: float = 5.0
    k: float = 4.0
    max_steps_in_episode: int = 200


@struct.dataclass
class EnvState(environment.EnvState):
    mjx_data: mjx.Data
    time: int


class SslEnv(environment.Environment[EnvState, EnvParams]):
    """Gymnax-compatible SSL MuJoCo environment."""

    def __init__(self):
        super().__init__()
        self.obs_dim = 12
        self.act_dim = 3

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @property
    def name(self) -> str:
        return "Ssl-v0"

    @property
    def num_actions(self) -> int:
        return self.act_dim

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        low = jnp.array([-jnp.inf, -jnp.inf, 0.0], dtype=jnp.float32)
        high = jnp.array([jnp.inf, jnp.inf, 1.0], dtype=jnp.float32)
        return spaces.Box(low, high, (self.act_dim,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-jnp.inf, jnp.inf, (self.obs_dim,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict({"time": spaces.Discrete(params.max_steps_in_episode)})

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        data = mjx.make_data(_MJX_MODEL)
        data = data.replace(qvel=data.qvel.at[BALL_QVEL_ADRS[0]].set(-1.0))
        state = EnvState(mjx_data=data, time=0)
        obs = self.get_obs(state, params)
        return obs, state

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, Dict[str, Any]]:
        data = state.mjx_data

        # parse action [vx, vy, kick]
        target_vel = action[:2]
        kick_flag = action[2] > 0.5

        # Kick logic
        robot_pos = data.qpos[ROBOT_QPOS_ADRS][:2]
        ball_pos = data.qpos[BALL_QPOS_ADRS][:2]
        rel = ball_pos - robot_pos
        dist = jnp.linalg.norm(rel)
        angle = jnp.arctan2(rel[1], rel[0])
        can_kick = (dist < 0.115) & (jnp.abs(angle) < 0.2) & kick_flag
        data = jax.lax.cond(
            can_kick,
            lambda d: d.replace(
                qvel=d.qvel.at[BALL_QVEL_ADRS[:2]].set((rel / dist) * 5.0)
            ),
            lambda d: d,
            data,
        )

        # Motion control
        vel = data.qvel[ROBOT_QVEL_ADRS][:2]
        err = target_vel - vel
        accel = params.k * err
        an = jnp.linalg.norm(accel)
        accel = jax.lax.cond(
            an > params.max_accel,
            lambda _: accel * (params.max_accel / an),
            lambda _: accel,
            operand=None,
        )
        forces = MASS * accel
        data = data.replace(ctrl=data.ctrl.at[ROBOT_CTRL_ADRS[:2]].set(forces))

        # physics step
        data = mjx.step(_MJX_MODEL, data)

        new_state = EnvState(mjx_data=data, time=state.time + 1)
        obs = self.get_obs(new_state, params)
        reward = -jnp.linalg.norm(TARGET_POS - obs[:2])
        done = self.is_terminal(new_state, params)
        info = {"discount": self.discount(new_state, params)}

        return obs, new_state, reward, done, info

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        data = state.mjx_data
        pos = data.qpos[ROBOT_QPOS_ADRS][:2]
        ori = data.qpos[ROBOT_QPOS_ADRS][2:3]
        vel = data.qvel[ROBOT_QVEL_ADRS][:2]
        ang = data.qvel[ROBOT_QVEL_ADRS][2:3]
        bpos = data.qpos[BALL_QPOS_ADRS][:3]
        bvel = data.qvel[BALL_QVEL_ADRS][:3]
        return jnp.concatenate([pos, ori, vel, ang, bpos, bvel], axis=0)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        return state.time >= params.max_steps_in_episode
