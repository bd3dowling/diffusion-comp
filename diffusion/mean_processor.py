"""Posterior mean processor definitions and registries."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array
from strenum import StrEnum

from diffusion.utils import extract_and_expand

__MODEL_MEAN_PROCESSOR__: dict[MeanProcessorType, type[MeanProcessor]] = {}


def register_mean_processor(name: MeanProcessorType):
    def wrapper(cls):
        if __MODEL_MEAN_PROCESSOR__.get(name):
            raise NameError(f"Name {name} is already registerd.")
        __MODEL_MEAN_PROCESSOR__[name] = cls
        return cls

    return wrapper


class MeanProcessorType(StrEnum):
    PREVIOUS = "previous"
    START = "start"
    EPSILON = "epsilon"


@dataclass
class MeanProcessor(ABC):
    betas: Array
    clip_denoised: bool

    def __post_init__(self):
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = jnp.append(1.0, self.alphas_cumprod[:-1])

        self.posterior_mean_coef1 = (
            self.betas * jnp.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * jnp.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    @classmethod
    def from_name(cls, name: MeanProcessorType, **kwargs) -> MeanProcessor:
        if not __MODEL_MEAN_PROCESSOR__.get(name):
            raise NameError(f"Name {name} is not defined.")
        return __MODEL_MEAN_PROCESSOR__[name](**kwargs)

    @abstractmethod
    def get_mean_and_xstart(self, x, t, model_output):
        raise NotImplementedError

    def process_xstart(self, x: Array):
        if self.clip_denoised:
            x = jnp.clip(x, -1, 1)
        return x


@register_mean_processor(name=MeanProcessorType.PREVIOUS)
class PreviousXMeanProcessor(MeanProcessor):
    def get_mean_and_xstart(self, x, t, model_output):
        mean = model_output
        pred_xstart = self.process_xstart(self.predict_xstart(x, t, model_output))
        return mean, pred_xstart

    def predict_xstart(self, x_t, t, x_prev):
        coef1 = extract_and_expand(1.0 / self.posterior_mean_coef1, t, x_t)
        coef2 = extract_and_expand(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t)
        return coef1 * x_prev - coef2 * x_t


@register_mean_processor(name=MeanProcessorType.START)
class StartXMeanProcessor(MeanProcessor):
    def get_mean_and_xstart(self, x, t, model_output):
        pred_xstart = self.process_xstart(model_output)
        mean = self.q_posterior_mean(x_start=pred_xstart, x_t=x, t=t)

        return mean, pred_xstart

    def q_posterior_mean(self, x_start, x_t, t):
        """Compute the mean of the diffusion posterior: q(x_{t-1} | x_t, x_0)"""
        coef1 = extract_and_expand(self.posterior_mean_coef1, t, x_start)
        coef2 = extract_and_expand(self.posterior_mean_coef2, t, x_t)

        return coef1 * x_start + coef2 * x_t


@register_mean_processor(name=MeanProcessorType.EPSILON)
class EpsilonXMeanProcessor(MeanProcessor):
    def __post_init__(self):
        super().__post_init__()

        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod - 1)

    def get_mean_and_xstart(self, x, t, model_output):
        pred_xstart = self.process_xstart(self.predict_xstart(x, t, model_output))
        mean = self.q_posterior_mean(pred_xstart, x, t)

        return mean, pred_xstart

    def q_posterior_mean(self, x_start, x_t, t):
        """Compute the mean of the diffusion posterior: q(x_{t-1} | x_t, x_0)"""
        coef1 = extract_and_expand(self.posterior_mean_coef1, t, x_start)
        coef2 = extract_and_expand(self.posterior_mean_coef2, t, x_t)
        return coef1 * x_start + coef2 * x_t

    def predict_xstart(self, x_t, t, eps):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, eps)
        return coef1 * x_t - coef2 * eps
