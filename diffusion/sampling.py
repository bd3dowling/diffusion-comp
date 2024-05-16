import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jaxtyping import Array, PRNGKeyArray


def sample_with_time(
    rng: PRNGKeyArray,
    n_samples: int,
    model: eqx.Module,
    alpha_bar: Array,
    beta: Array,
) -> tuple[Array, Array]:
    rng, step_rng = random.split(rng)
    all_outputs = np.zeros((len(beta) + 1, n_samples, 2))
    noised_data = random.normal(step_rng, (n_samples, 2))

    all_outputs[0, :, :] = noised_data

    for i in range(len(beta)):
        beta_i = beta[-i]
        alpha_bar_i = alpha_bar[-i] * jnp.ones((noised_data.shape[0], 1))
        noise_guess = jax.vmap(model)(noised_data, alpha_bar_i)
        rng, step_rng = random.split(rng)
        new_noise = random.normal(step_rng, noised_data.shape)
        noised_data: Array = (
            1
            / (1 - beta_i) ** 0.5
            * (noised_data - beta_i / (1 - alpha_bar_i) ** 0.5 * noise_guess)
        )
        if i < len(beta) - 1:
            noised_data += beta_i**0.5 * new_noise

        all_outputs[i + 1, :, :] = noised_data

    return noised_data, all_outputs
