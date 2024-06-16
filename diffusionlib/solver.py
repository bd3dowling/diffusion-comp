from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import StrEnum, auto
from typing import Callable, final

import jax.numpy as jnp
from jax import lax, random
from jaxtyping import Array, PRNGKeyArray

from diffusionlib.sde import SDE
from diffusionlib.util.misc import batch_mul

__SOLVER__: dict["SolverName", type["Solver"]] = {}


class SolverName(StrEnum):
    NONE = auto()
    EULER_MARUYAMA = auto()
    LANGEVIN_DYNAMICS = auto()


def register_solver(name: SolverName):
    def wrapper(cls):
        if __SOLVER__.get(name):
            raise NameError(f"Name {name} is already registered!")
        __SOLVER__[name] = cls

        return cls

    return wrapper

def get_solver(name: SolverName, **kwargs):
    if (solver_class := __SOLVER__.get(name)) is None:
        raise NameError(f"Name {name} is not defined!")

    class_fields = {field.name for field in fields(solver_class)}
    params = {field: value for field, value in kwargs.items() if field in class_fields}

    return solver_class(**params)


@dataclass(kw_only=True)
class Solver(ABC):
    num_steps: int = 0
    ts: Array = field(init=False)
    dt: Array = field(init=False)

    def __post_init__(self):
        self.ts, self.dt = _get_times(self.num_steps)

    @abstractmethod
    def prior_sampling(self, rng: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
        raise NotImplementedError

    @abstractmethod
    def update(self, rng: PRNGKeyArray, x: Array, t: Array) -> tuple[Array, Array]:
        raise NotImplementedError


@dataclass(kw_only=True)
class SDESolver(Solver, ABC):
    sde: SDE

    @final
    def prior_sampling(self, rng: Array, shape: tuple[int, ...]) -> Array:
        return self.sde.prior_sampling(rng, shape)

@register_solver(name=SolverName.NONE)
class NoneSolver(Solver):
    def update(self, rng, x, t):
        return x, x

    def prior_sampling(self, rng: Array, shape: tuple[int, ...]) -> Array:
        return random.uniform(rng, shape) # arbitrary


@register_solver(name=SolverName.EULER_MARUYAMA)
class EulerMaruyama(SDESolver):
    def update(self, rng, x, t):
        drift, diffusion = self.sde.step(x, t)

        z = random.normal(rng, x.shape)
        f = drift * self.dt
        g = diffusion * jnp.sqrt(self.dt)

        x_mean = x + f
        x = x_mean + batch_mul(g, z)

        return x, x_mean

@register_solver(name=SolverName.LANGEVIN_DYNAMICS)
@dataclass
class LangevinDynamics(SDESolver):
    snr: float
    score: Callable[[Array, Array], Array]

    def update(self, rng, x, t):
        rng, step_rng = random.split(rng)
        alpha = jnp.exp(2 * self.sde.marginal_log_mean_coeff(t))

        grad = self.score(x, t)
        grad_norm = jnp.linalg.norm(grad.reshape((grad.shape[0], -1)), axis=-1).mean()
        grad_norm = lax.pmean(grad_norm, axis_name="batch")

        noise = random.normal(step_rng, x.shape)
        noise_norm = jnp.linalg.norm(noise.reshape((noise.shape[0], -1)), axis=-1).mean()
        noise_norm = lax.pmean(noise_norm, axis_name="batch")

        dt = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + batch_mul(grad, dt)
        x = x_mean + batch_mul(2 * dt, noise)

        return x, x_mean


def _get_times(num_steps: int):
    # Defined in forward time, t \in [dt, 1.0], 0 < dt << 1
    ts, dt = jnp.linspace(0.0, 1.0, num_steps + 1, retstep=True)
    ts = ts[1:].reshape(-1, 1)
    return ts, dt
