"""Posterior variance processor definitions and registry."""

from abc import ABC, abstractmethod

import numpy as np
import torch

from diffusionlib.util.array import extract_and_expand

__MODEL_VAR_PROCESSOR__ = {}


def register_var_processor(name: str):
    def wrapper(cls):
        if __MODEL_VAR_PROCESSOR__.get(name, None):
            raise NameError(f"Name {name} is already registerd.")
        __MODEL_VAR_PROCESSOR__[name] = cls
        return cls

    return wrapper


def get_var_processor(name: str, **kwargs):
    if __MODEL_VAR_PROCESSOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __MODEL_VAR_PROCESSOR__[name](**kwargs)


class VarianceProcessor(ABC):
    @abstractmethod
    def __init__(self, betas):
        pass

    @abstractmethod
    def get_variance(self, x, t):
        pass


@register_var_processor(name="fixed_small")
class FixedSmallVarianceProcessor(VarianceProcessor):
    def __init__(self, betas):
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def get_variance(self, x, t):
        model_variance = self.posterior_variance
        model_log_variance = np.log(model_variance)

        model_variance = extract_and_expand(model_variance, t, x)
        model_log_variance = extract_and_expand(model_log_variance, t, x)

        return model_variance, model_log_variance


@register_var_processor(name="fixed_large")
class FixedLargeVarianceProcessor(VarianceProcessor):
    def __init__(self, betas):
        self.betas = betas

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def get_variance(self, x, t):
        model_variance = np.append(self.posterior_variance[1], self.betas[1:])
        model_log_variance = np.log(model_variance)

        model_variance = extract_and_expand(model_variance, t, x)
        model_log_variance = extract_and_expand(model_log_variance, t, x)

        return model_variance, model_log_variance


@register_var_processor(name="learned")
class LearnedVarianceProcessor(VarianceProcessor):
    def __init__(self, betas):
        pass

    def get_variance(self, x, t):
        model_log_variance = x
        model_variance = torch.exp(model_log_variance)
        return model_variance, model_log_variance


@register_var_processor(name="learned_range")
class LearnedRangeVarianceProcessor(VarianceProcessor):
    def __init__(self, betas):
        self.betas = betas

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])
        )

    def get_variance(self, x, t):
        model_var_values = x
        min_log = self.posterior_log_variance_clipped
        max_log = np.log(self.betas)

        min_log = extract_and_expand(min_log, t, x)
        max_log = extract_and_expand(max_log, t, x)

        # The model_var_values is [-1, 1] for [min_var, max_var]
        frac = (model_var_values + 1.0) / 2.0
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)
        return model_variance, model_log_variance
