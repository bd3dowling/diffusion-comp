from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import particles.distributions as dists
from jax import random
from jaxtyping import Array, PRNGKeyArray
from particles.core import SMC
from particles.state_space_models import Bootstrap, GuidedPF, StateSpaceModel

from diffusionlib.sampler.base import Sampler, SamplerName, register_sampler
from diffusionlib.sampler.ddim import DDIMVP


@register_sampler(name=SamplerName.SMC_DIFF_OPT)
@dataclass(kw_only=True)
class SMCDiffOptSampler(Sampler):
    base_sampler: DDIMVP
    obs_matrix: Array
    obs_noise: Array
    num_particles: int = 1000
    stack_samples: bool = False

    @property
    def dim_x(self) -> int:
        return self.obs_matrix.shape[1]

    @property
    def dim_y(self) -> int:
        return self.obs_matrix.shape[0]

    def sample(
        self, rng: PRNGKeyArray, x_0: Array | None = None, y: Array | None = None, **kwargs: Any
    ) -> Array:
        if y is None:
            raise ValueError("y must not be None")

        rng, sub_rng = random.split(rng)
        np.random.seed(sub_rng[0])

        data = _construct_obs_sequence(y, self.base_sampler)
        ssm = self.SSM(self.base_sampler, self.dim_x, self.dim_y, self.obs_matrix, self.obs_noise)
        fk = self.PF(ssm, data)
        smc = SMC(fk=fk, N=self.num_particles, store_history=True, **kwargs)
        smc.run()

        return smc.hist.X if self.stack_samples else smc.X

    class SSM(StateSpaceModel):
        def __init__(
            self, sampler: DDIMVP, dim_x: int, dim_y: int, a: Array, sigma_y: Array, **kwargs: Any
        ):
            super().__init__(**kwargs)
            self.sampler = sampler
            self.dim_x = dim_x
            self.dim_y = dim_y
            self.a = a
            self.sigma_y = sigma_y
            self.c_t = lambda t: self.sampler.sqrt_alphas_cumprod[-(t + 1)]
            self.d_t = lambda t: self.sampler.sqrt_1m_alphas_cumprod[-(t + 1)]
            self.ts = self.sampler.ts[::-1]

        def PX0(self):
            return dists.MvNormal(loc=jnp.zeros(self.dim_x))

        def PX(self, t, xp):
            vec_t = jnp.full(xp.shape[0], self.ts[t - 1])
            x_mean, std = self.sampler.posterior(xp, vec_t)
            scale = std[0]  # NOTE: std identical for all particles
            return dists.MvNormal(loc=x_mean, scale=scale)

        def PY(self, t, xp, x):
            cov = (self.c_t(t) ** 2 * self.sigma_y**2 * jnp.eye(self.dim_y)) + (
                self.d_t(t) ** 2 * self.a @ self.a.T
            )
            return dists.MvNormal(loc=x @ self.a.T, cov=cov)

    class PF(Bootstrap):
        def logG(self, t, xp, x):
            if t == 0:
                return super().logG(t, xp, x)

            return super().logG(t, xp, x) - self.ssm.PY(t - 1, xp, xp).logpdf(self.data[t - 1])


@register_sampler(name=SamplerName.MCG_DIFF)
@dataclass(kw_only=True)
class MCGDiffOptSampler(Sampler):
    base_sampler: DDIMVP
    obs_matrix: Array
    obs_noise: Array
    num_particles: int = 1000
    stack_samples: bool = False

    @property
    def dim_x(self) -> int:
        return self.obs_matrix.shape[1]

    @property
    def dim_y(self) -> int:
        return self.obs_matrix.shape[0]

    def sample(
        self, rng: PRNGKeyArray, x_0: Array | None = None, y: Array | None = None, **kwargs: Any
    ) -> Array:
        if y is None:
            raise ValueError("y must not be None")

        rng, sub_rng = random.split(rng)
        np.random.seed(sub_rng[0])

        data = _construct_obs_sequence(y, self.base_sampler)
        ssm = self.SSM(self.base_sampler, self.dim_x, self.dim_y, self.obs_matrix, self.obs_noise)
        fk = self.PF(ssm, data)
        smc = SMC(fk=fk, N=self.num_particles, store_history=True, **kwargs)
        smc.run()

        return smc.hist.X if self.stack_samples else smc.X

    class SSM(StateSpaceModel):
        def __init__(
            self, sampler: DDIMVP, dim_x: int, dim_y: int, a: Array, sigma_y: Array, **kwargs: Any
        ):
            super().__init__(**kwargs)
            self.sampler = sampler
            self.dim_x = dim_x
            self.dim_y = dim_y
            self.a = a
            self.sigma_y = sigma_y
            self.c_t = lambda t: self.sampler.sqrt_alphas_cumprod[-(t + 1)]
            self.d_t = lambda t: self.sampler.sqrt_1m_alphas_cumprod[-(t + 1)]
            self.ts = self.sampler.ts[::-1]

        def PX0(self):
            return dists.MvNormal(loc=jnp.zeros(self.dim_x))

        def PX(self, t, xp):
            return dists.MvNormal(loc=self.x_mean, scale=self.scale)

        def PY(self, t, xp, x):
            cov = (self.c_t(t) ** 2 * self.sigma_y**2 * jnp.eye(self.dim_y)) + (
                self.d_t(t) ** 2 * self.a @ self.a.T
            )
            return dists.MvNormal(loc=x @ self.a.T, cov=cov)

        def proposal0(self, data):
            return self.PX0()

        def proposal(self, t, xp, data):
            vec_t = jnp.full(xp.shape[0], self.ts[t - 1])

            x_mean, std = self.sampler.posterior(xp, vec_t)
            scale = std[0]  # NOTE: std identical for all particles

            # NOTE: Store so don't need to recompute when weighting
            self.x_mean, self.scale = x_mean, scale

            cov_1 = (
                self.c_t(t) ** 2 * self.sigma_y**2 * jnp.eye(self.dim_y)
                + self.d_t(t) ** 2 * self.a @ self.a.T
            )
            prec_1 = jnp.eye(self.dim_x) / (scale**2)
            prec_2_1 = self.a.T @ jnp.linalg.inv(cov_1)
            prec_2 = prec_2_1 @ self.a
            Sigma_star = jnp.linalg.inv(prec_1 + prec_2)

            mu_term_1 = x_mean @ prec_1.T
            mu_term_2 = prec_2_1 @ data[t]
            mu_star = (mu_term_1 + mu_term_2) @ Sigma_star.T

            return dists.MvNormal(loc=mu_star, cov=Sigma_star)

    class PF(GuidedPF):
        def logG(self, t, xp, x):
            if t == 0:
                return super().logG(t, xp, x)

            return super().logG(t, xp, x) - self.ssm.PY(t - 1, xp, xp).logpdf(self.data[t - 1])


def _construct_obs_sequence(init_obs: Array, base_sampler: DDIMVP) -> Array:
    y_s = init_obs * base_sampler.sqrt_alphas_cumprod[:, None]

    # NOTE: for particle filter since needs to run "backwards" (from T -> 0)
    return y_s[::-1]
