from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Type, final

import jax.numpy as jnp
from jax import random, vmap
from jaxtyping import Array, PRNGKeyArray

from diffusionlib.util.misc import batch_mul


class SDE(ABC):
    @abstractmethod
    def step(self, x: Array, t: Array) -> tuple[Array, Array]:
        raise NotImplementedError

    @abstractmethod
    def prior_sampling(self, rng: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
        raise NotImplementedError

    @abstractmethod
    def marginal_mean_coeff(self, t: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def marginal_variance(self, t: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def r2(self, t: Array, data_variance: Array) -> Array:
        """Get analytic variance of the distribution at time zero conditioned on x_t, given crude
        assumption thatthe data distribution is isotropic-Gaussian.

        Args:
            t (Array): Time index.
            data_variance (Array): Variance of data.

        Returns:
            Array: Analytic variance of distribution at time zero conditioned on x_t.
        """
        raise NotImplementedError

    def marginal_log_mean_coeff(self, t: Array) -> Array:
        return jnp.log(self.marginal_mean_coeff(t))

    def marginal_std(self, t: Array) -> Array:
        return jnp.sqrt(self.marginal_variance(t))

    @final
    def ratio(self, t: Array) -> Array:
        """Get the ratio of the marginal variance and mean coefficient.

        Args:
            t (Array): Time index.

        Returns:
            Array: Ratio of marginal mean variance and mean coefficient.
        """
        return self.marginal_variance(t) / self.marginal_mean_coeff(t)


@dataclass
class ForwardSDE(SDE, ABC):
    @property
    @abstractmethod
    def reverse_class(self) -> Type["ReverseSDE"]:
        raise NotImplementedError

    @final
    def reverse(self, score: Callable[[Array, Array], Array]) -> "ReverseSDE":
        return self.reverse_class(forward_sde=self, score=score)


@dataclass
class ReverseSDE(SDE, ABC):
    forward_sde: ForwardSDE
    score: Callable[[Array, Array], Array]

    @abstractmethod
    def get_estimate_x_0(
        self, observation_map: Callable[[Array], Array], shape: tuple[int, ...] | None = None
    ) -> Callable[[Array, Array], tuple[Array, tuple[Array, Array]]]:
        """Get a function returning the MMSE estimate of x_0|x_t.

        Args:
            observation_map (Callable[[Array], Array]): The observation map.
            shape (tuple[int, ...] | None, optional): Desired output shape. Defaults to None.

        Returns:
            Callable[[Array, Array], tuple[Array, tuple[Array, Array]]]: Function returning the
                MMSE estimate of x_0|x_t.
        """
        raise NotImplementedError

    @abstractmethod
    def get_estimate_x_0_vmap(
        self, observation_map: Callable[[Array], Array]
    ) -> Callable[[Array, Array], tuple[Array, tuple[Array, Array]]]:
        raise NotImplementedError

    @final
    def step(self, x: Array, t: Array) -> tuple[Array, Array]:
        drift, diffusion = self.forward_sde.step(x, t)
        drift = -drift + batch_mul(diffusion**2, self.score(x, t))
        return drift, diffusion

    @final
    def guide(self, guidance_score: Callable[[Array, Array], Array]) -> "ReverseSDE":
        return type(self)(forward_sde=self.forward_sde, score=guidance_score)

    @final
    def prior_sampling(self, rng: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
        return self.forward_sde.prior_sampling(rng, shape)

    @final
    def marginal_mean_coeff(self, t: Array) -> Array:
        return self.forward_sde.marginal_mean_coeff(t)

    @final
    def marginal_variance(self, t: Array) -> Array:
        return self.forward_sde.marginal_variance(t)

    @final
    def marginal_log_mean_coeff(self, t: Array) -> Array:
        return self.forward_sde.marginal_log_mean_coeff(t)

    @final
    def r2(self, t: Array, data_variance: Array) -> Array:
        return self.forward_sde.r2(t, data_variance)

    def marginal_std(self, t: Array) -> Array:
        return self.forward_sde.marginal_std(t)


@dataclass
class VP(ForwardSDE):
    beta_min: Array = field(default_factory=lambda: jnp.array(0.1))
    beta_max: Array = field(default_factory=lambda: jnp.array(20.0))

    @property
    def reverse_class(self) -> Type["RVP"]:
        return RVP

    def step(self, x: Array, t: Array) -> tuple[Array, Array]:
        beta_t = self._beta(t)
        drift = -0.5 * batch_mul(beta_t, x)
        diffusion = jnp.sqrt(beta_t)

        return drift, diffusion

    def marginal_mean_coeff(self, t: Array) -> Array:
        return jnp.exp(self.marginal_log_mean_coeff(t))

    def marginal_log_mean_coeff(self, t: Array) -> Array:
        return -0.5 * t * self.beta_min - 0.25 * t**2 * (self.beta_max - self.beta_min)

    def marginal_variance(self, t: Array) -> Array:
        return 1.0 - jnp.exp(2 * self.marginal_log_mean_coeff(t))

    def prior_sampling(self, rng: Array, shape: tuple[int, ...]) -> Array:
        return random.normal(rng, shape)

    def r2(self, t: Array, data_variance: Array) -> Array:
        alpha = jnp.exp(2 * self.marginal_log_mean_coeff(t))
        return (1 - alpha) * data_variance / (1 - alpha + alpha * data_variance)

    def _beta(self, t: Array) -> Array:
        return self.beta_min + t * (self.beta_max - self.beta_min)


@dataclass
class VE(ForwardSDE):
    sigma_min: Array = field(default_factory=lambda: jnp.array(0.01))
    sigma_max: Array = field(default_factory=lambda: jnp.array(378.0))

    @property
    def reverse_class(self) -> Type["RVE"]:
        return RVE

    def step(self, x: Array, t: Array) -> tuple[Array, Array]:
        drift = jnp.zeros_like(x)
        diffusion = self._sigma(t) * jnp.sqrt(
            2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min))
        )

        return drift, diffusion

    def marginal_mean_coeff(self, t: Array) -> Array:
        return jnp.ones_like(t)

    def marginal_log_mean_coeff(self, t: Array) -> Array:
        return jnp.zeros_like(t)

    def marginal_variance(self, t: Array) -> Array:
        return self._sigma(t) ** 2

    def marginal_std(self, t: Array) -> Array:
        return self._sigma(t)

    def prior_sampling(self, rng: Array, shape: tuple[int, ...]) -> Array:
        return random.normal(rng, shape) * self.sigma_max

    def r2(self, t: Array, data_variance: Array) -> Array:
        variance = self.marginal_variance(t)
        return variance * data_variance / (variance + data_variance)

    def _sigma(self, t: Array) -> Array:
        log_sigma_min = jnp.log(self.sigma_min)
        log_sigma_max = jnp.log(self.sigma_max)

        return jnp.exp(log_sigma_min + t * (log_sigma_max - log_sigma_min))


@dataclass()
class RVP(ReverseSDE):
    def get_estimate_x_0(
        self, observation_map: Callable[[Array], Array], shape: tuple[int, ...] | None = None
    ) -> Callable[[Array, Array], tuple[Array, tuple[Array, Array]]]:
        batch_observation_map = vmap(observation_map)

        def estimate_x_0(x: Array, t: Array) -> tuple[Array, tuple[Array, Array]]:
            m_t = self.marginal_mean_coeff(t)
            v_t = self.marginal_variance(t)
            s = self.score(x, t)
            x_0 = batch_mul(x + batch_mul(v_t, s), 1.0 / m_t)

            return (
                (batch_observation_map(x_0.reshape(shape)), (s, x_0))
                if shape
                else (batch_observation_map(x_0), (s, x_0))
            )

        return estimate_x_0

    def get_estimate_x_0_vmap(
        self, observation_map: Callable[[Array], Array]
    ) -> Callable[[Array, Array], tuple[Array, tuple[Array, Array]]]:
        def estimate_x_0(x, t):
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)

            m_t = self.marginal_mean_coeff(t)
            v_t = self.marginal_variance(t)
            s = self.score(x, t)
            x_0 = (x + v_t * s) / m_t
            return observation_map(x_0), (s, x_0)

        return estimate_x_0


@dataclass
class RVE(ReverseSDE):
    def get_estimate_x_0(
        self, observation_map: Callable[[Array], Array], shape: tuple[int, ...] | None = None
    ) -> Callable[[Array, Array], tuple[Array, tuple[Array, Array]]]:
        batch_observation_map = vmap(observation_map)

        def estimate_x_0(x: Array, t: Array) -> tuple[Array, tuple[Array, Array]]:
            v_t = self.marginal_variance(t)
            s = self.score(x, t)
            x_0 = x + batch_mul(v_t, s)

            return (
                (batch_observation_map(x_0.reshape(shape)), (s, x_0))
                if shape
                else (batch_observation_map(x_0), (s, x_0))
            )

        return estimate_x_0

    def get_estimate_x_0_vmap(
        self, observation_map: Callable[[Array], Array]
    ) -> Callable[[Array, Array], tuple[Array, tuple[Array, Array]]]:
        def estimate_x_0(x, t):
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)

            v_t = self.marginal_variance(t)
            s = self.score(x, t)
            x_0 = x + v_t * s

            return observation_map(x_0), (s, x_0)

        return estimate_x_0
