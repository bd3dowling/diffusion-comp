from functools import partial
from typing import Any

import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from flax.linen import Module
from jax import Array, jit
from optax import OptState, Params, GradientTransformation


class FullyConnectedWithTime(Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""

    @nn.compact
    def __call__(self, x: Array, t: int) -> Array:
        in_size = x.shape[1]
        n_hidden = 256
        t = jnp.concatenate(
            [t - 0.5, jnp.cos(2 * jnp.pi * t), jnp.sin(2 * jnp.pi * t), -jnp.cos(4 * jnp.pi * t)],
            axis=1,
        )
        x = jnp.concatenate([x, t], axis=1)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x


def loss(
    params: Params,
    model: Module,
    rng: Array,
    data: Array,
    alpha_bar: Array,
) -> Array:
    rng, step_rng = random.split(rng)
    r_alpha_bar = random.choice(step_rng, alpha_bar, (data.shape[0], 1))

    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, data.shape)

    noised_data = data * r_alpha_bar**0.5 + noise * (1 - r_alpha_bar) ** 0.5

    output = model.apply(params, noised_data, r_alpha_bar)
    loss = jnp.mean((noise - output) ** 2)

    return loss


def fit(
    n_epochs: int,
    params: dict[str, Any],
    optimizer: GradientTransformation,
    rng: Array,
    model: Module,
    batch: Array,
    alpha_bar: Array,
) -> Params:
    opt_state = optimizer.init(params)
    losses: list[Any] = []

    @partial(jit, static_argnums=[3])
    def step(
        params: Params,
        opt_state: OptState,
        rng: Array,
        model: Module,
        batch: Array,
        alpha_bar: Array,
    ) -> tuple[Params, OptState, Any]:
        loss_value, grads = jax.value_and_grad(loss)(params, model, rng, batch, alpha_bar)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss_value

    for i in range(n_epochs):
        rng, step_rng = random.split(rng)
        params, opt_state, loss_value = step(params, opt_state, step_rng, model, batch, alpha_bar)
        losses.append(loss_value)

        if (i + 1) % 5_000 == 0:
            mean_loss = np.mean(np.array(losses))
            print("Epoch %d,\t Loss %f " % (i + 1, mean_loss))
            losses.clear()

    return params
