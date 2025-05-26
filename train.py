from functools import partial

import distrax
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from flax.nnx.nn.initializers import constant, orthogonal
from jax.sharding import PartitionSpec as P

print(jax.local_devices())
NUM_DEVICES = jax.local_device_count()

mesh = jax.make_mesh((NUM_DEVICES,), ("x",))

SEED = 7
ITERATIONS = 50
NUM_STEPS = 1024
NUM_ENVS = 2048
VALUE_COEF = 0.5
ENTROPY_COEF = 0.001
BATCH_SIZE = 4096
K = 5
EPS = 0.1
DISCOUNT_FACTOR = 0.99
GAE_LAMBDA = 0.95

ENV_NAME = "CartPole-v1"
OPTIMIZER = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adamw(1e-3),
)

assert NUM_ENVS % NUM_DEVICES == 0, "for parallel experience collection, NUM_ENVS should be divisible by NUM_DEVICES!"
NUM_ENVS_PER_DEVICE = NUM_ENVS // NUM_DEVICES

env, env_params = gymnax.make(ENV_NAME)
vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))


class Model(nnx.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_actions,
        dropout=0.1,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.policy = nnx.Sequential(
            nnx.Linear(
                input_dim,
                hidden_dim,
                rngs=rngs,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            ),
            nnx.Dropout(dropout, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                hidden_dim,
                hidden_dim,
                rngs=rngs,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            ),
            nnx.Dropout(dropout, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                hidden_dim,
                num_actions,
                rngs=rngs,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            ),
        )
        self.value = nnx.Sequential(
            nnx.Linear(
                input_dim,
                hidden_dim,
                rngs=rngs,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            ),
            nnx.Dropout(dropout, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                hidden_dim,
                hidden_dim,
                rngs=rngs,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            ),
            nnx.Dropout(dropout, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                hidden_dim,
                1,
                rngs=rngs,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
            ),
        )

    def __call__(self, x: jax.Array) -> tuple[distrax.Categorical, jax.Array]:
        logits = self.policy(x)
        return distrax.Categorical(logits), self.value(x)


def get_action_logprobs_value(model: Model, obs: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    pi, value = model(obs)
    action = pi.sample(seed=key)
    return action, pi.log_prob(action), value


def get_value(model: Model, obs: jax.Array) -> jax.Array:
    return model(obs)[1]


@nnx.jit
def get_action(model: Model, obs: jax.Array, key: jax.Array) -> jax.Array:
    return model(obs)[0].sample(seed=key)


def loss_fn(model, observations, actions, values, actions_log_probs, advantages, returns):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # standardize advantages

    pi, value = model(observations)
    value = value.flatten()

    prob_ratio = jnp.exp(pi.log_prob(actions) - actions_log_probs)  # = p_new(a_t) / p_old(a_t) because of log rules log(a/b) = log(a) - log(b)

    policy_loss = -jnp.mean(
        jnp.minimum(
            prob_ratio * advantages,
            jnp.clip(prob_ratio, min=1.0 - EPS, max=1.0 + EPS) * advantages,
        )
    )  # policy_loss is -L_CLIP because we want to maximize L_CLIP but using gradient descent
    value_loss = jnp.mean(jnp.square(value - returns))
    entropy_loss = -jnp.mean(pi.entropy())
    return policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss


def calculate_gae_returns(
    rewards,  # (T, N)
    values,  # (T, N)
    dones,  # (T, N)
    last_value,  # (N,)
):
    """adapted from https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py#L142"""

    def _get_advantages(gae_and_next_value, reward_value_done):
        gae, next_value = gae_and_next_value
        reward, value, done = reward_value_done
        delta = reward + DISCOUNT_FACTOR * next_value * (1 - done) - value
        gae = delta + DISCOUNT_FACTOR * GAE_LAMBDA * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (
            jnp.zeros_like(last_value),  # gae_t = delta_t
            last_value,  # v_(t+1) is last_value
        ),
        (rewards, values, dones),
        reverse=True,  # scanning the trajectory batch in reverse order
    )
    return advantages, advantages + values  # advantages + values = returns


def step_once(carry, _):
    key, model_state, env_state, last_observation = carry
    model = nnx.merge(*model_state)

    key, subkey = jax.random.split(key)
    action, log_prob, value = get_action_logprobs_value(model, last_observation, subkey)

    # Automatic env resetting in gymnax step!
    observation, env_state, reward, done, _ = vmap_step(jax.random.split(subkey, NUM_ENVS_PER_DEVICE), env_state, action, env_params)

    return (key, model_state, env_state, observation), (
        last_observation,
        value.flatten(),
        action,
        log_prob,
        reward,
        jnp.where(done, 1, 0),
    )


@partial(jax.shard_map, mesh=mesh, in_specs=(P("x"), None), out_specs=(P("x"), P("x")), check_vma=False)
def collect_experience(keys, model_state):
    key = keys[0]  # keys was (1,)
    observation, env_state = vmap_reset(jax.random.split(key, NUM_ENVS_PER_DEVICE), env_params)
    env_state = jax.lax.pvary(env_state, "x")

    (key, _, _, last_observation), trajectory = jax.lax.scan(step_once, (key, model_state, env_state, observation), length=NUM_STEPS)
    return last_observation, jax.tree.map(lambda x: x.swapaxes(0, 1), trajectory)  # (NUM_ENVS_PER_DEVICE,), (NUM_ENVS_PER_DEVICE, NUM_STEPS, *)


def iter_once(carry, i):
    key, model_state, optimizer_state = carry
    key, subkey = jax.random.split(key)
    jax.debug.print("iteration {i}", i=i)

    # Run policy πθold in environment for T timesteps
    per_device_key = jax.random.split(subkey, NUM_DEVICES)
    last_observation, trajectory = collect_experience(
        per_device_key, model_state
    )  # (NUM_DEVICES * NUM_ENVS_PER_DEVICE,), (NUM_DEVICES * NUM_ENVS_PER_DEVICE, NUM_STEPS, *)

    trajectory = jax.tree.map(lambda x: x.swapaxes(0, 1), trajectory)  # (NUM_STEPS, NUM_ENVS, *)
    observations, values, actions, log_probs, rewards, dones = trajectory  # (NUM_STEPS, NUM_ENVS, *)

    last_value = get_value(nnx.merge(*model_state), last_observation).flatten()  # one more required for gae's deltas

    # Compute advantage estimates Â_1,..., Â_T
    advantages, returns = calculate_gae_returns(rewards, values, dones, last_value)  # (T, N)

    # flatten NUM_ENVS * NUM_STEPS of experience
    (
        observations,
        values,
        actions,
        log_probs,
        rewards,
        dones,
        advantages,
        returns,
    ) = jax.tree.map(
        lambda x: x.flatten() if len(x.shape) == 2 else x.reshape(NUM_STEPS * NUM_ENVS, -1),
        (*trajectory, advantages, returns),
    )  # (NUM_ENVS * NUM_STEPS, *)

    jax.debug.print("iter mean return: {r}", r=jnp.mean(returns))

    # Optimize surrogate L wrt θ, with K epochs and minibatch size M ≤NT
    def one_epoch(carry, epoch):
        key, model_state, optimizer_state = carry
        jax.debug.print("epoch {epoch}", epoch=epoch)

        key, subkey = jax.random.split(key)
        idxs = jax.random.permutation(subkey, NUM_ENVS * NUM_STEPS)
        batches_idxs = idxs.reshape(-1, BATCH_SIZE)

        def one_batch(carry, batch_idxs):
            model_state, optimizer_state = carry
            model = nnx.merge(*model_state)

            loss, grads = nnx.value_and_grad(loss_fn)(
                model,
                observations[batch_idxs],
                actions[batch_idxs],
                values[batch_idxs],
                log_probs[batch_idxs],
                advantages[batch_idxs],
                returns[batch_idxs],
            )

            # pure optax update
            current_params = nnx.state(model, nnx.Param)
            updates, optimizer_state = OPTIMIZER.update(grads, optimizer_state, current_params)
            new_params = optax.apply_updates(current_params, updates)
            nnx.update(model, new_params)

            return (nnx.split(model), optimizer_state), None

        (model_state, optimizer_state), _ = jax.lax.scan(one_batch, (model_state, optimizer_state), batches_idxs)

        return (key, model_state, optimizer_state), None

    (key, model_state, optimizer_state), _ = jax.lax.scan(one_epoch, (key, model_state, optimizer_state), jnp.arange(K))
    return (key, model_state, optimizer_state), None


if __name__ == "__main__":
    key = jax.random.key(SEED)
    model = Model(
        env.observation_space(env_params).shape[0], 128, env.action_space(env_params).n
    )  # Acrobot has obs (6,) and action (1,) in [0,3] range
    optimizer_state = OPTIMIZER.init(nnx.state(model, nnx.Param))
    (key, model_state, optimizer_state), _ = jax.lax.scan(iter_once, (key, nnx.split(model), optimizer_state), jnp.arange(ITERATIONS))
    model = nnx.merge(*model_state)

    # EVAL
    import gymnasium as gym

    env = gym.make(ENV_NAME, render_mode="human")
    observation, _ = env.reset(seed=42)
    for _ in range(1000):
        key, subkey = jax.random.split(key)
        action = get_action(model, observation, subkey).item()
        observation, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()

    env.close()

