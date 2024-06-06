"""Posterior variance processor definitions and registries."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array
from strenum import StrEnum

from diffusion.utils import extract_and_expand

__MODEL_VAR_PROCESSOR__: dict[VarianceProcessorType, type[VarianceProcessor]] = {}


def register_var_processor(name: VarianceProcessorType):
    def wrapper(cls):
        if __MODEL_VAR_PROCESSOR__.get(name):
            raise NameError(f"Name {name} is already registerd.")
        __MODEL_VAR_PROCESSOR__[name] = cls
        return cls

    return wrapper


class VarianceProcessorType(StrEnum):
    FIXED_SMALL = "fixed_small"
    FIXED_LARGE = "fixed_large"
    LEARNED = "learned"
    LEARNED_RANGE = "learned_range"


@dataclass
class VarianceProcessor(ABC):
    betas: Array
    posterior_variance: Array
    posterior_log_variance_clipped: Array

    @classmethod
    def from_name(cls, name: VarianceProcessorType, **kwargs) -> VarianceProcessor:
        if not __MODEL_VAR_PROCESSOR__.get(name):
            raise NameError(f"Name {name} is not defined.")
        return __MODEL_VAR_PROCESSOR__[name](**kwargs)

    @abstractmethod
    def get_variance(self, x, t: int):
        raise NotImplementedError


@register_var_processor(name=VarianceProcessorType.FIXED_SMALL)
class FixedSmallVarianceProcessor(VarianceProcessor):
    def get_variance(self, x, t):
        model_variance = self.posterior_variance
        model_log_variance = jnp.log(model_variance)

        model_variance = extract_and_expand(model_variance, t, x)
        model_log_variance = extract_and_expand(model_log_variance, t, x)

        return model_variance, model_log_variance


@register_var_processor(name=VarianceProcessorType.FIXED_LARGE)
class FixedLargeVarianceProcessor(VarianceProcessor):
    def get_variance(self, x, t):
        model_variance = jnp.append(self.posterior_variance[1], self.betas[1:])
        model_log_variance = jnp.log(model_variance)

        model_variance = extract_and_expand(model_variance, t, x)
        model_log_variance = extract_and_expand(model_log_variance, t, x)

        return model_variance, model_log_variance


@register_var_processor(name=VarianceProcessorType.LEARNED)
class LearnedVarianceProcessor(VarianceProcessor):
    def get_variance(self, x, t):
        model_log_variance = x
        model_variance = jnp.exp(model_log_variance)
        return model_variance, model_log_variance


@register_var_processor(name=VarianceProcessorType.LEARNED_RANGE)
class LearnedRangeVarianceProcessor(VarianceProcessor):
    def get_variance(self, x, t):
        model_var_values = x
        min_log = self.posterior_log_variance_clipped
        max_log = jnp.log(self.betas)

        min_log = extract_and_expand(min_log, t, x)
        max_log = extract_and_expand(max_log, t, x)

        # The model_var_values is [-1, 1] for [min_var, max_var]
        frac = (model_var_values + 1.0) / 2.0
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = jnp.exp(model_log_variance)
        return model_variance, model_log_variance
