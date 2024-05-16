import jax.numpy as jnp
import jax.random as random
from jax import Array


def get_alpha_beta(n: int, beta_min: float = 0.1, beta_max: float = 20.0) -> tuple[Array, Array]:
    beta = jnp.array([beta_min / n + i / (n * (n - 1)) * (beta_max - beta_min) for i in range(n)])

    return 1 - beta, beta


def forward_noising(
    data: Array,
    n: int,
    alpha_bar: Array,
    rng_key: Array | None = None,
) -> tuple[Array, Array]:
    rng_key = random.PRNGKey(0) if not rng_key else rng_key

    train_samples = jnp.repeat(jnp.reshape(data, (1, *data.shape)), n, axis=0)

    noise = jnp.reshape(
        random.normal(rng_key, (data.shape[0] * n, data.shape[1])),
        (n, *data.shape),
    )

    sqrt_alpha_bar = jnp.reshape(jnp.sqrt(alpha_bar), (-1, 1, 1))
    sqrt_one_m_alpha_bar = jnp.reshape(jnp.sqrt(1 - alpha_bar), (-1, 1, 1))

    noisy_image = sqrt_alpha_bar * train_samples + sqrt_one_m_alpha_bar * noise

    return noisy_image, noise
