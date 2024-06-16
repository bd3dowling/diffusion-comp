import jax.numpy as jnp
import jax.random as random
import numpy as np
from scipy.stats import wasserstein_distance


def sliced_wasserstein(rng, dist_1, dist_2, n_slices=100):
    projections = random.normal(rng, (n_slices, dist_1.shape[1]))
    projections = projections / jnp.linalg.norm(projections, axis=-1)[:, None]
    dist_1_projected = projections @ dist_1.T
    dist_2_projected = projections @ dist_2.T
    return np.mean(
        [
            wasserstein_distance(u_values=d1, v_values=d2)
            for d1, d2 in zip(dist_1_projected, dist_2_projected)
        ]
    )
