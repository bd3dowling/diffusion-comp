from typing import Any

import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns
from jax import Array, jit, vmap
from matplotlib.animation import FuncAnimation
from matplotlib.image import AxesImage

CM = sns.color_palette("mako_r", as_cmap=True)


def sample_sphere(num_points: int) -> Array:
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1 / num_points), num_points)
    xs = jnp.cos(alphas)
    ys = jnp.sin(alphas)

    return jnp.stack([xs, ys], axis=1)


def plot_heatmap(positions: Array, area_min: float = -2.0, area_max: float = 2.0) -> AxesImage:
    # positions: locations of all particles in R^2, array (J, 2)
    # area_min: lowest x and y coordinate
    # area_max: highest x and y coordinate

    # will plot a heatmap of all particles in the area [area_min, area_max] x [area_min, area_max]

    heatmap_data = _get_heatmap_data(positions, area_min, area_max)
    extent = (area_min, area_max, area_max, area_min)
    im = plt.imshow(heatmap_data, cmap=CM, interpolation="nearest", extent=extent)
    ax = plt.gca()
    ax.invert_yaxis()
    return im


def animate_heatmap(samples: Array, all_outputs: Array, **kwargs: Any) -> FuncAnimation:
    _im = plot_heatmap(samples)
    _fig = plt.figure(figsize=(8, 8))

    def _animate(frame: int) -> list[AxesImage]:
        _im.set_data(_get_heatmap_data(all_outputs[frame, :, :], **kwargs))
        return [_im]

    return animation.FuncAnimation(_fig, _animate, frames=all_outputs.shape[0])


@jit
def _get_heatmap_data(positions: Array, area_min: float = -2.0, area_max: float = 2.0) -> Array:
    v_kernel = vmap(_small_kernel, in_axes=(0, None, None))
    return jnp.sum(v_kernel(positions, area_min, area_max), axis=0)


def _small_kernel(z: Array, area_min: float, area_max: float) -> Array:
    a = jnp.linspace(area_min, area_max, 512)
    x, y = jnp.meshgrid(a, a)
    dist = (x - z[0]) ** 2 + (y - z[1]) ** 2
    return jnp.exp(-350 * dist)
