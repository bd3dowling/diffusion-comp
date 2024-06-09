"""Plotting code for the examples."""
from functools import partial

import jax.numpy as jnp
import jax.random as random
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy
from jax import jit, vmap
from scipy.stats import wasserstein_distance

BG_ALPHA = 1.0
MG_ALPHA = 1.0
FG_ALPHA = 0.3
color_posterior = "#a2c4c9"
color_algorithm = "#ff7878"
dpi_val = 1200
cmap = "magma"


def plot_heatmap(samples, area_bounds, lengthscale=350.0):
    """Plots a heatmap of all samples in the area area_bounds x area_bounds.
    Args:
      samples: locations of particles shape (num_particles, 2)
    """

    def small_kernel(z, area_bounds):
        a = jnp.linspace(area_bounds[0], area_bounds[1], 512)
        x, y = jnp.meshgrid(a, a)
        dist = (x - z[0]) ** 2 + (y - z[1]) ** 2
        hm = jnp.exp(-lengthscale * dist)
        return hm

    # jit most of the code, but use the helper functions since cannot jit all of it because of plt
    @jit
    def produce_heatmap(samples, area_bounds):
        return jnp.sum(vmap(small_kernel, in_axes=(0, None))(samples, area_bounds), axis=0)

    fig, ax = plt.subplots()
    hm = produce_heatmap(samples, area_bounds)
    ax.imshow(hm, interpolation="nearest", extent=area_bounds + area_bounds)
    ax.invert_yaxis()
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")

    plt.close()

    return fig


def image_grid(x, image_size, num_channels):
    img = x.reshape(-1, image_size, image_size, num_channels)
    w = int(np.sqrt(img.shape[0]))
    return (
        img.reshape((w, w, image_size, image_size, num_channels))
        .transpose((0, 2, 1, 3, 4))
        .reshape((w * image_size, w * image_size, num_channels))
    )


def plot_samples(x, image_size=32, num_channels=3, fname="samples"):
    img = image_grid(x, image_size, num_channels)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(img, cmap=cmap)
    plt.savefig(fname + ".png", bbox_inches="tight", pad_inches=0.0)
    # plt.savefig(fname + '.pdf', bbox_inches='tight', pad_inches=0.0)
    plt.close()


def plot_scatter(samples, index, fname="samples", lims=None):
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(BG_ALPHA)
    ax.scatter(samples[:, index[0]], samples[:, index[1]], color="red", label=r"$x$")
    ax.legend()
    ax.set_xlabel(rf"$x_{index[0]}$")
    ax.set_ylabel(rf"$x_{index[1]}$")
    if lims is not None:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.draw()
    fig.savefig(fname + ".png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_samples_1D(samples, image_size, x_max=5.0, fname="samples 1D", alpha=FG_ALPHA):
    x = np.linspace(-x_max, x_max, image_size)
    plt.plot(x, samples[..., 0].T, alpha=alpha)
    plt.xlim(-5.0, 5.0)
    plt.ylim(-5.0, 5.0)
    plt.savefig(fname + ".png")
    plt.close()


def plot_animation(fig, ax, animate, frames, fname, fps=20, bitrate=800, dpi=300):
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=1, fargs=(ax,))
    # Set up formatting for the movie files
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=fps, metadata=dict(artist="Me"), bitrate=bitrate)
    # Note that mp4 does not work on pdf
    ani.save(f"{fname}.mp4", writer=writer, dpi=dpi)


def plot_score(score, scaler, t, area_bounds=[-3.0, 3.0]):
    # this helper function is here so that we can jit
    # We can not jit the whole function since plt.quiver cannot be jitted
    @partial(jit, static_argnums=[0])
    def helper(score, t, area_bounds):
        x = jnp.linspace(area_bounds[0], area_bounds[1], 16)
        x, y = jnp.meshgrid(x, x)
        grid = jnp.stack([x.flatten(), y.flatten()], axis=1)
        t = jnp.ones((grid.shape[0],)) * t
        scores = score(scaler(grid), t)
        return grid, scores

    fig, ax = plt.subplots()
    grid, scores = helper(score, t, area_bounds)
    ax.quiver(grid[:, 0], grid[:, 1], scores[:, 0], scores[:, 1])
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")
    ax.set_aspect("equal", adjustable="box")

    plt.close()

    return fig


def plot_temperature_schedule(sde, solver):
    """Plots the temperature schedule of the SDE marginals.

    Args:
      sde: a valid SDE class.
    """
    m2 = sde.mean_coeff(solver.ts) ** 2
    v = sde.variance(solver.ts)
    plt.plot(solver.ts, m2, label="m2")
    plt.plot(solver.ts, v, label="v")
    plt.legend()
    plt.savefig("plot_temperature_schedule.png")
    plt.close()


def plot_single_image(
    noise_std, dim, dim_y, timesteps, i, name, indices, samples, color=color_algorithm
):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(*samples[:, indices].T, alpha=0.5, color=color, edgecolors="black", rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-22, 22])
    ax.set_ylim([-22, 22])
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
    plt.savefig(
        f"inverse_problem_comparison_out_dist_{noise_std}_{dim}_{dim_y}_{timesteps}_{i}_{name}.pdf",
        dpi=dpi_val,
    )
    plt.savefig(
        f"inverse_problem_comparison_out_dist_{noise_std}_{dim}_{dim_y}_{timesteps}_{i}_{name}.png",
        transparent=True,
        dpi=dpi_val,
    )
    plt.close(fig)


def plot_image(
    noise_std, dim, dim_y, timesteps, i, name, indices, diffusion_samples, target_samples=None
):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(
        *target_samples[:, indices].T,
        alpha=0.5,
        color=color_posterior,
        edgecolors="black",
        rasterized=True,
    )
    ax.scatter(
        *diffusion_samples[:, indices].T,
        alpha=0.5,
        color=color_algorithm,
        edgecolors="black",
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-22, 22])
    ax.set_ylim([-22, 22])
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
    plt.savefig(
        f"inverse_problem_comparison_out_dist_{noise_std}_{dim}_{dim_y}_{timesteps}_{i}_{name}.pdf",
        dpi=dpi_val,
    )
    plt.savefig(
        f"inverse_problem_comparison_out_dist_{noise_std}_{dim}_{dim_y}_{timesteps}_{i}_{name}.png",
        transparent=True,
        dpi=dpi_val,
    )
    plt.close(fig)


def sliced_wasserstein(rng, dist_1, dist_2, n_slices=100):
    projections = random.normal(rng, (n_slices, dist_1.shape[1]))
    projections = projections / jnp.linalg.norm(projections, axis=-1)[:, None]
    dist_1_projected = projections @ dist_1.T
    dist_2_projected = projections @ dist_2.T
    return np.mean(
        [
            wasserstein_distance(u_values=d1, v_values=d2)
            for d1, d2 in zip(dist_1_projected, dist_2_projected)
        ]
    )


def Wasserstein2(m1, C1, m2, C2):
    C1_half = scipy.linalg.sqrtm(C1)
    C_half = jnp.asarray(
        np.asarray(np.real(scipy.linalg.sqrtm(C1_half @ C2 @ C1_half)), dtype=float)
    )
    return jnp.linalg.norm(m1 - m2) ** 2 + jnp.trace(C1) + jnp.trace(C2) - 2 * jnp.trace(C_half)


def Distance2(m1, C1, m2, C2):
    C2_half = jnp.linalg.cholesky(C2)
    C1_half = jnp.linalg.cholesky(C1)
    return jnp.linalg.norm(m1 - m2) ** 2 + jnp.linalg.norm(C1_half - C2_half) ** 2


def plot(train_data, test_data, mean, variance, fname="plot.png"):
    X, y = train_data
    X_show, f_show, variance_show = test_data
    # Plot result
    fig, ax = plt.subplots(1, 1)
    ax.plot(X_show, f_show, label="True", color="orange")
    ax.plot(X_show, mean, label="Prediction", linestyle="--", color="blue")
    ax.scatter(X, y, label="Observations", color="black", s=20)
    ax.fill_between(
        X_show.flatten(),
        mean - 2.0 * jnp.sqrt(variance),
        mean + 2.0 * jnp.sqrt(variance),
        alpha=FG_ALPHA,
        color="blue",
    )
    ax.fill_between(
        X_show.flatten(),
        f_show - 2.0 * jnp.sqrt(variance_show),
        f_show + 2.0 * jnp.sqrt(variance_show),
        alpha=FG_ALPHA,
        color="orange",
    )
    ax.set_xlim((X_show[0], X_show[-1]))
    ax.set_ylim((-2.4, 2.4))
    ax.grid(visible=True, which="major", linestyle="-")
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(BG_ALPHA)
    ax.patch.set_alpha(MG_ALPHA)
    ax.legend()
    fig.savefig(fname)
    plt.close()


def plot_beta_schedule(sde, solver):
    """Plots the temperature schedule of the SDE marginals.

    Args:
        sde: a valid SDE class.
    """
    beta_t = sde.beta_min + solver.ts * (sde.beta_max - sde.beta_min)
    diffusion = jnp.sqrt(beta_t)

    plt.plot(solver.ts, beta_t, label="beta_t")
    plt.plot(solver.ts, diffusion, label="diffusion_t")
    plt.legend()
    plt.savefig("plot_beta_schedule.png")
    plt.close()
