"""Operator/task definitions and registry."""

from abc import ABC, abstractmethod
from functools import partial
from typing import final, overload

import torch
from jaxtyping import Array
from strenum import StrEnum
from torch import Tensor
from torch.nn import functional as F

from diffusionlib.util.image import Blurkernel, Resizer
from external.motionblur.motionblur import Kernel

__OPERATOR__ = {}

JaxOrTorchArray = Array | Tensor


class OperatorName(StrEnum):
    NOISE = "noise"
    SUPER_RESOLUTION = "super_resolution"
    MOTION_BLUR = "motion_blur"
    GAUSSIAN_BLUR = "gaussian_blur"
    INPAINTING = "inpainting"


def register_operator(name: OperatorName):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls

    return wrapper


def get_operator(name: OperatorName, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @overload
    def forward(self, data: torch.Tensor, **kwargs) -> Tensor:
        ...

    @overload
    def forward(self, data: Array, **kwargs) -> Array:
        ...

    @final
    def forward(self, data: JaxOrTorchArray, **kwargs) -> JaxOrTorchArray:
        # calculate A * X
        if isinstance(data, Array):
            return self._jax_forward(data, **kwargs)
        elif isinstance(data, Tensor):
            return self._torch_forward(data, **kwargs)
        else:
            raise TypeError(f"Unsupported type: {type(data)=}")

    @overload
    def transpose(self, data: Tensor, **kwargs) -> Tensor:
        ...

    @overload
    def transpose(self, data: Array, **kwargs) -> Array:
        ...

    @final
    def transpose(self, data: JaxOrTorchArray, **kwargs) -> JaxOrTorchArray:
        # calculate A^T * X
        if isinstance(data, Array):
            return self._jax_transpose(data, **kwargs)
        elif isinstance(data, Tensor):
            return self._torch_transpose(data, **kwargs)
        else:
            raise TypeError(f"Unsupported type: {type(data)=}")

    @overload
    def ortho_project(self, data: Tensor, **kwargs) -> Tensor:
        ...

    @overload
    def ortho_project(self, data: Array, **kwargs) -> Array:
        ...

    def ortho_project(self, data: JaxOrTorchArray, **kwargs) -> JaxOrTorchArray:
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    @overload
    def project(self, data: Tensor, measurement: Tensor, **kwargs) -> Tensor:
        ...

    @overload
    def project(self, data: Array, measurement: Array, **kwargs) -> Array:
        ...

    def project(
        self, data: JaxOrTorchArray, measurement: JaxOrTorchArray, **kwargs
    ) -> JaxOrTorchArray:
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)

    @abstractmethod
    def _jax_forward(self, data: Array, **kwargs) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _torch_forward(self, data: torch.Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def _jax_transpose(self, data: Array, **kwargs) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _torch_transpose(self, data: torch.Tensor, **kwargs) -> Tensor:
        raise NotImplementedError


@register_operator(name=OperatorName.NOISE)
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device

    def _jax_forward(self, data: Array, **kwargs) -> Array:
        return data

    def _jax_transpose(self, data: Array, **kwargs) -> Array:
        return data

    def _torch_forward(self, data: Tensor, **kwargs) -> Tensor:
        return data

    def _torch_transpose(self, data: Tensor, **kwargs) -> Tensor:
        return data


@register_operator(name=OperatorName.SUPER_RESOLUTION)
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1 / scale_factor).to(device)

    def _torch_forward(self, data: Tensor, **kwargs) -> Tensor:
        return self.down_sample(data)

    def _torch_transpose(self, data: Tensor, **kwargs) -> Tensor:
        return self.up_sample(data)

    def _jax_forward(self, data: Array, **kwargs) -> Array:
        raise NotImplementedError

    def _jax_transpose(self, data: Array, **kwargs) -> Array:
        raise NotImplementedError

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)


@register_operator(name=OperatorName.MOTION_BLUR)
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(
            blur_type="motion", kernel_size=kernel_size, std=intensity, device=device
        ).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)

    def _torch_forward(self, data: Tensor, **kwargs) -> Tensor:
        # A^T * A
        return self.conv(data)

    def _torch_transpose(self, data: Tensor, **kwargs) -> Tensor:
        return data

    def _jax_forward(self, data: Array, **kwargs) -> Array:
        raise NotImplementedError

    def _jax_transpose(self, data: Array, **kwargs) -> Array:
        raise NotImplementedError

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name=OperatorName.GAUSSIAN_BLUR)
class GaussianBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(
            blur_type="gaussian", kernel_size=kernel_size, std=intensity, device=device
        ).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def _torch_forward(self, data: Tensor, **kwargs) -> Tensor:
        return self.conv(data)

    def _torch_transpose(self, data: Tensor, **kwargs) -> Tensor:
        return data

    def _jax_forward(self, data: Array, **kwargs) -> Array:
        raise NotImplementedError

    def _jax_transpose(self, data: Array, **kwargs) -> Array:
        raise NotImplementedError

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name=OperatorName.INPAINTING)
class InpaintingOperator(LinearOperator):
    """This operator get pre-defined mask and return masked image."""

    def __init__(self, device):
        self.device = device

    def _torch_forward(self, data: Tensor, **kwargs) -> Tensor:
        mask = kwargs.get("mask")
        if not mask:
            raise ValueError("Inpainting operator requires a mask")
        return data * mask

    def _torch_transpose(self, data: Tensor, **kwargs) -> Tensor:
        return data

    def _jax_forward(self, data: Array, **kwargs) -> Array:
        raise NotImplementedError

    def _jax_transpose(self, data: Array, **kwargs) -> Array:
        raise NotImplementedError

    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)
