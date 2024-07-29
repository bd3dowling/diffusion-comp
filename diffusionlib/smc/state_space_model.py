from typing import Any

import jax.numpy as jnp
import particles.distributions as dists
from jaxtyping import Array
from particles.state_space_models import StateSpaceModel

from diffusionlib.sampler.ddim import DDIMVP


# Define state space model
class SimpleObservationSSM(StateSpaceModel):
    def __init__(self, sampler: DDIMVP, dim_x: int, a: Array, sigma_y: Array, **kwargs: Any):
        super().__init__(**kwargs)
        self.sampler = sampler
        self.dim_x = dim_x
        self.a = a
        self.sigma_y = sigma_y

    def PX0(self):
        # TODO: check if can swap np for jnp...
        return dists.MvNormal(loc=jnp.zeros(self.dim_x))

    def PX(self, t, xp):
        # NOTE: sampler runs in reverse (T -> 0) hence the `-t`.
        # That is, parameter `t` here is actually `T-t` for our sampling purposes...
        vec_t = jnp.full(xp.shape[0], self.sampler.ts[-t])
        x_mean, std = self.sampler.posterior(xp, vec_t)

        # std is just scalar which should be applied to identity covariance for each sample.
        # particles expects either a single scalar value or a (N, d) array to apply individually.
        # hence we duplicate the value d times to align with this.
        std_full = jnp.repeat(std[:, jnp.newaxis], self.dim_x, axis=1)

        return dists.MvNormal(loc=x_mean, scale=std_full)

    def PY(self, t, xp, x):
        return dists.MvNormal(loc=x @ self.a.T, cov=self.sigma_y)

    def proposal0(self, data):
        return self.PX0()

    def proposal(self, t, xp, data):
        return self.PX(t, xp)
