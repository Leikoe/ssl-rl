import time
from typing import NamedTuple

import jax
import mujoco
from jax import numpy as jnp
from mujoco import mjx

# mujoco consts
_MJ_MODEL = mujoco.MjModel.from_xml_path("scene.xml")
_MJX_MODEL = mjx.put_model(_MJ_MODEL)
ROBOT_QPOS_ADRS = jnp.array(
    [
        _MJ_MODEL.joint("x_pos").qposadr[0],
        _MJ_MODEL.joint("y_pos").qposadr[0],
        _MJ_MODEL.joint("orientation").qposadr[0],
    ]
)  # (x, y, orienation)
ROBOT_QVEL_ADRS = jnp.array(
    [
        _MJ_MODEL.joint("x_pos").dofadr[0],
        _MJ_MODEL.joint("y_pos").dofadr[0],
        _MJ_MODEL.joint("orientation").dofadr[0],
    ]
)  # (vx, vy, vangular)
ROBOT_CTRL_ADRS = jnp.array(
    [
        _MJ_MODEL.actuator("forward_motor").id,
        _MJ_MODEL.actuator("left_motor").id,
        _MJ_MODEL.actuator("orientation_motor").id,
    ]
)  # (fx, fy, fangular) motors
BALL_QPOS_ADRS = (
    jnp.arange(3) + _MJ_MODEL.joint("free_ball").qposadr[0]
)  # (x, y, z) pos
BALL_QVEL_ADRS = (
    jnp.arange(6) + _MJ_MODEL.joint("free_ball").dofadr[0]
)  # linear speed, rotational speed
M = _MJ_MODEL.body("robot_base").mass

# temp goal
TARGET_POS = jnp.array([3, 2.0])


class Action(NamedTuple):
    target_vel: jax.Array
    kick: bool


class Observation(NamedTuple):
    pos: jax.Array  # (x, y)
    vel: jax.Array  # (vx, vy)
    orientation: jax.Array  # (orientation,) angle in radians
    angular_vel: jax.Array  # (angular_vel,) radians/s
    ball_pos: jax.Array  # (x, y, z)
    ball_vel: jax.Array  # (vx, vy, vz)


class State(NamedTuple):
    mjx_data: mjx.Data
    observation: Observation
    reward: float
    terminated: bool


class Ssl:
    def __init__(self, max_accel=5.0, k=4.0):
        """
        Initializes a Ssl env factory.
        """

        self.max_accel: float = max_accel
        self.k: float = k

    def init(self, key: jax.Array) -> State:
        mjx_data = mjx.make_data(_MJX_MODEL)
        # init code here ...
        mjx_data = mjx_data.replace(
            qvel=mjx_data.qvel.at[BALL_QVEL_ADRS[0]].set(-1.0)
        )  # TODO: fix kick orientation
        # if key is not None:
        #     pos = jax.random.uniform(
        #         key, (2,), float, jnp.array([-4.0, -3.0]), jnp.array([-0.2, 3.0])
        #     )
        #     mjx_data = mjx_data.replace(
        #         qpos=mjx_data.qpos.at[ROBOT_QPOS_ADRS[:2]].set(pos)
        #     )

        obs = self._get_obs(mjx_data)
        return State(
            mjx_data=mjx_data,
            observation=obs,
            reward=jnp.linalg.norm(TARGET_POS - obs.pos),
            terminated=False,
        )

    def step(self, state: State, action: Action, key: jax.Array) -> State:
        """
        Steps the env using the given `action`.

        Args:
            state: The env state to step.
            action: The action to take.
            key: The key used for randomness within the step

        Returns:
            Env: The new env state.
            Observation: The observation of the new env state.
            float: The reward.
            bool: terminated (reached a final pos, good or bad).
            # bool: truncated (was forcefully terminated).
        """
        del key
        mjx_data = state.mjx_data
        new_qvel = mjx_data.qvel

        # kick
        robot_pos = mjx_data.qpos[ROBOT_QPOS_ADRS][:2]
        ball_pos = mjx_data.qpos[BALL_QPOS_ADRS][:2]

        robot_to_ball = ball_pos - robot_pos
        robot_to_ball_angle = jnp.arctan2(robot_to_ball[1], robot_to_ball[0])
        robot_to_ball_distance = jnp.linalg.norm(robot_to_ball)
        robot_to_ball_normalized = robot_to_ball / robot_to_ball_distance

        REACH = 0.09 + 0.025  # robot radius + ball radius
        kick_would_hit_ball = (robot_to_ball_distance < REACH) & (
            (robot_to_ball_angle < 0.2) & (robot_to_ball_angle > -0.2)
        )
        new_qvel = jax.lax.select(
            jnp.logical_and(
                action.kick, kick_would_hit_ball
            ),  # if we want to kick and the kick can hit the ball, apply vel
            new_qvel.at[BALL_QVEL_ADRS[:2]].set(robot_to_ball_normalized * 5.0),
            new_qvel,
        )
        mjx_data = mjx_data.replace(qvel=new_qvel)

        vel = mjx_data.qvel[ROBOT_QVEL_ADRS][:2]

        target_vel = action.target_vel  # for now
        vel_err = target_vel - vel

        # clip speed
        a_target = self.k * vel_err
        a_target_norm = jnp.linalg.norm(a_target)
        a_target = jax.lax.select(
            jnp.linalg.norm(a_target) > self.max_accel,
            a_target * (self.max_accel / a_target_norm),
            a_target,
        )

        # compute force
        f = M * a_target

        print(state.observation.pos.shape)
        mjx_data = mjx_data.replace(ctrl=mjx_data.ctrl.at[ROBOT_CTRL_ADRS[:2]].set(f))
        mjx_data = mjx.step(_MJX_MODEL, mjx_data)
        obs = self._get_obs(mjx_data)
        reward = -jnp.linalg.norm(TARGET_POS - obs.pos)
        return State(
            mjx_data=mjx_data, observation=obs, reward=reward, terminated=False
        )
        # return State(mjx_data=mjx_data, observation=state.observation, reward=state.reward, terminated=state.terminated)

    def _get_obs(self, mjx_data: mjx.Data) -> Observation:
        return Observation(
            pos=mjx_data.qpos[ROBOT_QPOS_ADRS][:2],
            orientation=mjx_data.qpos[ROBOT_QPOS_ADRS][2],
            vel=mjx_data.qvel[ROBOT_QVEL_ADRS][:2],
            angular_vel=mjx_data.qvel[ROBOT_QVEL_ADRS][2],
            ball_pos=mjx_data.qpos[BALL_QPOS_ADRS][:3],
            ball_vel=mjx_data.qvel[BALL_QVEL_ADRS][:3],
        )


if __name__ == "__main__":
    N_ENVS = 128

    key = jax.random.key(0)

    ssl = Ssl()
    jitted_vmapped_init = jax.jit(jax.vmap(ssl.init))
    jitted_vmapped_step = jax.jit(jax.vmap(ssl.step))
    jitted_vmapped_norm = jax.jit(jax.vmap(jnp.linalg.norm))

    envs: State = jitted_vmapped_init(jax.random.split(key, N_ENVS))

    mj_datas: list[mujoco.MjData] = mjx.get_data(_MJ_MODEL, envs.mjx_data)

    duration = 2.0  # (seconds)
    framerate = 25  # (Hz)

    frames = []
    renderer = mujoco.Renderer(_MJ_MODEL, width=720, height=480)
    while mj_datas[0].time < duration:
        print(f"step {len(frames)} ({mj_datas[0].time})")
        step_start = time.time()

        robots_xy = envs.observation.pos
        balls_xy = envs.observation.ball_pos[:, :2]

        robot_to_ball = balls_xy - robots_xy
        robot_to_ball_angle = jnp.arctan2(robot_to_ball[:, 1], robot_to_ball[:, 0])
        robot_to_ball_distance = jitted_vmapped_norm(robot_to_ball)

        kick = robot_to_ball_distance < (0.09 + 0.025)

        target_vels = jnp.array([[1.0, 0.0]], dtype=jnp.float32).repeat(
            N_ENVS, axis=0
        )  # placeholder for policy output
        # print(target_vels)
        actions = Action(target_vels, kick)

        # print(actions.target_vel.shape, actions.kick.shape)

        envs: State = jitted_vmapped_step(
            envs, actions, None
        )  # jax.random.split(key, N_ENVS))
        # print(envs.observation, envs.reward)
        mj_datas = mjx.get_data(_MJ_MODEL, envs.mjx_data)
        if len(frames) < mj_datas[0].time * framerate:
            renderer.update_scene(mj_datas[0])
            pixels = renderer.render()
            frames.append(pixels)
    renderer.close()

    from PIL import Image

    imgs = [Image.fromarray(img) for img in frames]
    # duration is the number of milliseconds between frames; this is 25 fps
    imgs[0].save(
        "render.gif", save_all=True, append_images=imgs[1:], duration=40, loop=0
    )
