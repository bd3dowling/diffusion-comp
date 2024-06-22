from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, Callable

import jax.numpy as jnp
from jax import lax, random, vmap
from jaxtyping import Array, PRNGKeyArray

from diffusionlib.solver import NoneSolver, Solver, SolverName, get_solver
from diffusionlib.util.misc import (
    batch_mul,
    continuous_to_discrete,
    get_linear_beta_function,
    get_times,
    get_timestep,
)

# NOTE: unused for now...
DDIM_METHODS = ("pigdmvp", "pigdmve", "ddimve", "ddimvp", "kgdmvp", "kgdmve", "stsl")

__SAMPLER__: dict["SamplerName", type["Sampler"]] = {}


class SamplerName(StrEnum):
    PREDICTOR_CORRECTOR = auto()
    DDIM_VP = auto()
    DDIM_VE = auto()


def register_sampler(name: SamplerName):
    def wrapper(cls):
        if __SAMPLER__.get(name):
            raise NameError(f"Name {name} is already registered!")
        __SAMPLER__[name] = cls
        return cls

    return wrapper


def get_sampler(name: SamplerName) -> type["Sampler"]:
    if __SAMPLER__.get(name) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SAMPLER__[name]


class Sampler(ABC):
    @abstractmethod
    def sample(self, rng: PRNGKeyArray, x_0: Array | None = None) -> Array:
        raise NotImplementedError


@register_sampler(name=SamplerName.PREDICTOR_CORRECTOR)
@dataclass
class PCSampler(Sampler):
    shape: tuple[int, ...]
    outer_solver: Solver
    inner_solver: Solver = field(default_factory=lambda: NoneSolver())
    denoise: bool = True
    stack_samples: bool = False
    inverse_scaler: Callable[[Array], Array] = lambda x: x
    _has_inner_solver: bool = field(init=False)

    def __post_init__(self):
        self._has_inner_solver = not isinstance(self.inner_solver, NoneSolver)

    @classmethod
    def from_solver_names(
        cls,
        *,
        shape: tuple[int, ...],
        outer_solver_name: SolverName,
        outer_solver_kwargs: dict[str, Any],
        inner_solver_name: SolverName = SolverName.NONE,
        inner_solver_kwargs: dict[str, Any] | None = None,
        denoise: bool = True,
        stack_samples: bool = False,
        inverse_scaler: Callable[[Array], Array] = lambda x: x,
    ):
        if inner_solver_kwargs is None:
            inner_solver_kwargs = {}

        outer_solver = get_solver(outer_solver_name, **outer_solver_kwargs)
        inner_solver = get_solver(inner_solver_name, **inner_solver_kwargs)

        return cls(
            shape=shape,
            outer_solver=outer_solver,
            inner_solver=inner_solver,
            denoise=denoise,
            stack_samples=stack_samples,
            inverse_scaler=inverse_scaler,
        )

    @property
    def num_function_evaluations(self) -> int:
        return jnp.size(self._outer_ts) * (jnp.size(self._inner_ts) + 1)

    def sample(self, rng: PRNGKeyArray, x_0: Array | None = None) -> Array:
        rng, step_rng = random.split(rng)

        if x_0 is None:
            x_0 = (
                self.inner_solver.prior_sampling(step_rng, self.shape)
                if self._has_inner_solver
                else self.outer_solver.prior_sampling(step_rng, self.shape)
            )
        else:
            assert x_0.shape == self.shape

        x = x_0
        (_, x, x_mean), xs = lax.scan(self._outer_step, (rng, x, x), self._outer_ts, reverse=True)

        return_samples = (
            self.inverse_scaler(xs)
            if self.stack_samples
            else self.inverse_scaler(x_mean if self.denoise else x)
        )

        return return_samples

    def _outer_step(
        self, carry: tuple[PRNGKeyArray, Array, Array], outer_t: Array
    ) -> tuple[tuple[PRNGKeyArray, Array, Array], Array]:
        rng, x, x_mean = carry
        vec_t = jnp.full(self.shape[0], outer_t)
        rng, step_rng = random.split(rng)
        x, x_mean = self.outer_solver.update(step_rng, x, vec_t)

        if self._has_inner_solver:
            (rng, x, x_mean, vec_t), _ = lax.scan(
                self._inner_step, (step_rng, x, x_mean, vec_t), self._inner_ts
            )

        return ((rng, x, x_mean), x_mean) if self.denoise else ((rng, x, x_mean), x)

    def _inner_step(
        self, carry: tuple[PRNGKeyArray, Array, Array, Array], inner_t: Array
    ) -> tuple[tuple[PRNGKeyArray, Array, Array, Array], Array]:
        rng, x, x_mean, vec_t = carry
        rng, step_rng = random.split(rng)
        x, x_mean = self.inner_solver.update(step_rng, x, vec_t)

        return (rng, x, x_mean, vec_t), inner_t

    @property
    def _outer_ts(self) -> Array:
        return self.outer_solver.ts

    @property
    def _inner_ts(self) -> Array:
        return self.inner_solver.ts


@dataclass(kw_only=True)
class DDIMVP(Sampler):
    """DDIM Markov chain. For the DDPM Markov Chain or VP SDE."""

    # model: DDIM parameterizes the `epsilon(x, t) = -1. * fwd_marginal_std(t) * score(x, t)` function.
    # eta: the hyperparameter for DDIM, a value of `eta=0.0` is deterministic 'probability ODE' solver, `eta=1.0` is DDPMVP.
    num_steps: int = 1000
    shape: tuple[int, ...]
    model: Callable[[Array, Array], Array]
    beta_min: float
    beta_max: float
    eta: float
    denoise: bool = True
    stack_samples: bool = False
    inverse_scaler: Callable[[Array], Array] = lambda x: x

    def __post_init__(self):
        self.ts, self.dt = get_times(self.num_steps)
        self.beta, _ = get_linear_beta_function(beta_min=0.1, beta_max=20.0)
        self.discrete_betas = continuous_to_discrete(vmap(self.beta)(self.ts.flatten()), self.dt)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        self.alphas_cumprod_prev = jnp.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = jnp.sqrt(self.alphas_cumprod_prev)
        self.sqrt_1m_alphas_cumprod_prev = jnp.sqrt(1.0 - self.alphas_cumprod_prev)

    def sample(self, rng: PRNGKeyArray, x_0: Array | None = None) -> Array:
        rng, step_rng = random.split(rng)

        if x_0 is None:
            x_0 = self._prior_sampling(step_rng, self.shape)
        else:
            assert x_0.shape == self.shape

        x = x_0
        (_, x, x_mean), xs = lax.scan(self._step, (rng, x, x), self.ts, reverse=True)

        return_samples = (
            self.inverse_scaler(xs)
            if self.stack_samples
            else self.inverse_scaler(x_mean if self.denoise else x)
        )

        return return_samples

    def _prior_sampling(self, rng: Array, shape: tuple[int, ...]) -> Array:
        return random.normal(rng, shape)

    def _step(
        self, carry: tuple[PRNGKeyArray, Array, Array], outer_t: Array
    ) -> tuple[tuple[PRNGKeyArray, Array, Array], Array]:
        rng, x, x_mean = carry
        rng, step_rng = random.split(rng)

        vec_t = jnp.full(self.shape[0], outer_t)
        x, x_mean = self._update(step_rng, x, vec_t)

        return ((rng, x, x_mean), x_mean) if self.denoise else ((rng, x, x_mean), x)

    def _update(self, rng: Array, x: Array, t: Array) -> tuple[Array, Array]:
        x_mean, std = self._posterior(x, t)
        z = random.normal(rng, x.shape)
        x = x_mean + batch_mul(std, z)
        return x, x_mean

    def _posterior(self, x: Array, t: Array) -> tuple[Array, Array]:
        """Get parameters for $p(x_{t-1} \mid x_t)$"""
        epsilon = self.model(x, t)
        timestep = get_timestep(t, self.ts[0], self.ts[-1], self.num_steps)
        m = self.sqrt_alphas_cumprod[timestep]
        sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
        v = sqrt_1m_alpha**2
        alpha_cumprod = self.alphas_cumprod[timestep]
        alpha_cumprod_prev = self.alphas_cumprod_prev[timestep]
        m_prev = self.sqrt_alphas_cumprod_prev[timestep]
        v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep] ** 2
        x_0 = batch_mul((x - batch_mul(sqrt_1m_alpha, epsilon)), 1.0 / m)
        coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha_cumprod / alpha_cumprod_prev))
        coeff2 = jnp.sqrt(v_prev - coeff1**2)
        x_mean = batch_mul(m_prev, x_0) + batch_mul(coeff2, epsilon)
        std = coeff1

        return x_mean, std

    def _get_estimate_x_0(self, observation_map, clip=False, centered=True):
        # NOTE: if not clip, value won't matter; for typing purposes
        a_min, a_max = (-1.0, 1.0) if (clip and centered) else (0.0, 1.0)

        batch_observation_map = vmap(observation_map)

        def estimate_x_0(x, t, timestep):
            m = self.sqrt_alphas_cumprod[timestep]
            sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
            epsilon = self.model(x, t)
            x_0 = batch_mul(x - batch_mul(sqrt_1m_alpha, epsilon), 1.0 / m)
            if clip:
                x_0 = jnp.clip(x_0, a_min=a_min, a_max=a_max)
            return batch_observation_map(x_0), (epsilon, x_0)

        return estimate_x_0
