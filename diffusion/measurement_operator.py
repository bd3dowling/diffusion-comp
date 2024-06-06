from abc import ABC, abstractmethod

from jaxtyping import Array


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data: Array, **kwargs) -> Array:
        # calculate A * X
        raise NotImplementedError

    @abstractmethod
    def transpose(self, data: Array, **kwargs):
        # calculate A^T * X
        raise NotImplementedError

    def ortho_project(self, data: Array, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


class NoiseOperator(LinearOperator):
    def forward(self, data, **kwargs):
        return data

    def transpose(self, data, **kwargs):
        return data

    def ortho_project(self, data, **kwargs):
        return data

    def project(self, data, measurement, **kwargs):
        return data
