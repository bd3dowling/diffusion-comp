"""Utility functions related to Bayesian inversion."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum, auto
from typing import Callable, Type

import jax.numpy as jnp
from jax import grad
from jaxtyping import Array

from diffusionlib._sde.score_sde import SDE

__CONDITIONING_METHOD__: dict["ConditioningMethodName", Type["ConditioningMethod"]] = {}


class ConditioningMethodName(StrEnum):
    NONE = auto()
    DIFFUSION_POSTERIOR_SAMPLING = auto()
    DIFFUSION_POSTERIOR_SAMPLING_MOD = auto()
    PSEUDO_INVERSE_GUIDANCE = auto()
    VJP_GUIDANCE = auto()
    VJP_GUIDANCE_ALT = auto()
    VJP_GUIDANCE_MASK = auto()
    VJP_GUIDANCE_DIAG = auto()
    JAC_REV_GUIDANCE = auto()
    JAC_REV_GUIDANCE_DIAG = auto()
    JAC_FWD_GUIDANCE = auto()
    JAC_FWD_GUIDANCE_DIAG = auto()


def register_conditioning_method(name: ConditioningMethodName):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls

    return wrapper


def get_conditioning_method(name: ConditioningMethodName, **kwargs):
    if (cond_method_class := __CONDITIONING_METHOD__.get(name)) is None:
        raise NameError(f"Name {name} is not defined!")

    class_fields = {field.name for field in fields(cond_method_class)}
    params = {field: value for field, value in kwargs.items() if field in class_fields}

    return cond_method_class(**params)


@dataclass
class ConditioningMethod(ABC):
    sde: SDE
    y: Array

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not is_dataclass(cls):
            raise TypeError(f"Class {cls.__name__} must be a dataclass")

    @property
    @abstractmethod
    def guidance_score_func(self) -> Callable[[Array, float], Array]:
        raise NotImplementedError


@register_conditioning_method(ConditioningMethodName.DIFFUSION_POSTERIOR_SAMPLING)
@dataclass
class DPS(ConditioningMethod):
    """
    Implementation of score guidance suggested in
    `Diffusion Posterior Sampling for general noisy inverse problems'
    Chung et al. 2022,
    https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/guided_diffusion/condition_methods.py

    Computes a single (batched) gradient.

    NOTE: This is not how Chung et al. 2022 implemented their method, but is a related
    continuous time method.

    Args:
        scale: Hyperparameter of the method.
            See https://arxiv.org/pdf/2209.14687.pdf#page=20&zoom=100,144,757
    """

    observation_map: Callable[[Array], Array]
    scale: float

    @property
    def guidance_score_func(self) -> Callable[[Array, float], Array]:
        def get_l2_norm(y, estimate_h_x_0):
            def l2_norm(x, t):
                h_x_0, (s, _) = estimate_h_x_0(x, t)
                innovation = y - h_x_0
                return jnp.linalg.norm(innovation), s

            return l2_norm

        estimate_h_x_0 = self.sde.get_estimate_x_0(self.observation_map)
        l2_norm = get_l2_norm(self.y, estimate_h_x_0)
        likelihood_score = grad(l2_norm, has_aux=True)

        def guidance_score(x, t):
            ls, s = likelihood_score(x, t)
            gs = s - self.scale * ls
            return gs

        return guidance_score
