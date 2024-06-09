from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array

from diffusionlib.measurement_operator import LinearOperator, NoiseOperator


@dataclass
class PosteriorSampling:
    operator: LinearOperator = NoiseOperator()
    scale: float = 1.0

    def conditioning(self, x_prev, x_t, x_0_hat, y, **kwargs):
        norm_grad = self.norm_function(x_0_hat, y)
        return x_t - self.scale * norm_grad

    @jax.grad
    def norm_function(self, x: Array, y: Array, **kwargs) -> Array:
        return jnp.linalg.norm(y - self.operator.forward(x, **kwargs))
