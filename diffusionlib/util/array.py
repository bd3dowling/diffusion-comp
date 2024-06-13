"""Utility functions for working with numpy, jax and torch arrays/tensors."""


import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array


def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)


def extract_and_expand_jax(array: Array, time: Array, target: Array):
    array = array[time].astype(jnp.float32)

    # Expand dimensions to match the target array's number of dimensions
    while array.ndim < target.ndim:
        array = array[..., jnp.newaxis]  # Using jnp.newaxis to add a new last dimension

    # Use broadcasting to expand the array dimensions to match the target
    expanded_array = jnp.broadcast_to(array, target.shape)

    return expanded_array


def expand_as(array, target):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    elif isinstance(array, np.floating):
        array = torch.tensor([array])

    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)

    return array.expand_as(target).to(target.device)


def to_numpy(x):
    x = x.detach().cpu().squeeze().numpy()
    return np.clip(np.transpose(x, (1, 2, 0)), a_min=0.0, a_max=1.0)
