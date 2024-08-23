from abc import ABC, abstractmethod
from enum import StrEnum, auto
from typing import Any

from jaxtyping import Array, PRNGKeyArray

# NOTE: unused for now...
DDIM_METHODS = ("pigdmvp", "pigdmve", "ddimve", "ddimvp", "kgdmvp", "kgdmve", "stsl")

__SAMPLER__: dict["SamplerName", type["Sampler"]] = {}


class SamplerName(StrEnum):
    PREDICTOR_CORRECTOR = auto()
    DDIM_VP = auto()
    DDIM_VE = auto()
    SMC_DIFF_OPT = auto()
    MCG_DIFF = auto()


def register_sampler(name: SamplerName):
    def wrapper(cls):
        if __SAMPLER__.get(name):
            raise NameError(f"Name {name} is already registered!")
        __SAMPLER__[name] = cls
        return cls

    return wrapper


def get_sampler(name: SamplerName, **kwargs) -> "Sampler":
    if __SAMPLER__.get(name) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SAMPLER__[name](**kwargs)


class Sampler(ABC):
    @abstractmethod
    def sample(
        self, rng: PRNGKeyArray, x_0: Array | None = None, y: Array | None = None, **kwargs: Any
    ) -> Array:
        raise NotImplementedError
