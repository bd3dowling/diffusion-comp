from dataclasses import dataclass, field
from typing import Any, Callable

import jax.numpy as jnp
from jax import lax, random
from jaxtyping import Array, PRNGKeyArray

from diffusionlib.sampler.base import Sampler, SamplerName, register_sampler
from diffusionlib.solver import NoneSolver, Solver, SolverName, get_solver


@register_sampler(name=SamplerName.PREDICTOR_CORRECTOR)
@dataclass(kw_only=True)
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
