from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array

import diffusionlib.conditioning_method.jax_port as jax_port
from diffusionlib.operator.jax import LinearOperator, NoiseOperator


@dataclass
class PosteriorSampling:
    operator: LinearOperator = NoiseOperator()
    scale: float = 1.0

    def conditioning(self, x_prev, x_t, x_0_hat, y, **kwargs):
        norm_grad = self.norm_function(x_0_hat, y)
        return x_t - self.scale * norm_grad

    @jax_port.grad
    def norm_function(self, x: Array, y: Array, **kwargs) -> Array:
        return jnp.linalg.norm(y - self.operator.forward(x, **kwargs))
