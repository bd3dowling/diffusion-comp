"""Solver classes, including Markov chains."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random
from jax import grad, jacrev, lax, vjp, vmap
from jax.lax import scan
from jaxtyping import Array, PRNGKeyArray

from diffusionlib.conditioning_method import ConditioningMethodName, get_conditioning_method
from diffusionlib.mean_processor import MeanProcessor, MeanProcessorType
from diffusionlib.util.array import extract_and_expand
from diffusionlib.util.misc import (
    batch_mul,
    batch_mul_A,
    continuous_to_discrete,
    get_linear_beta_function,
    get_sigma_function,
    get_times,
    get_timestep,
    shared_update,
)
from diffusionlib.variance_processor import VarianceProcessor, VarianceProcessorType


def get_sampler(
    shape, outer_solver, inner_solver=None, denoise=True, stack_samples=False, inverse_scaler=None
):
    """Get a sampler from (possibly interleaved) numerical solver(s).

    Args:
      shape: Shape of array, x. (num_samples,) + obj_shape, where x_shape is the shape
        of the object being sampled from, for example, an image may have
        obj_shape==(H, W, C), and so shape==(N, H, W, C) where N is the number of samples.
      outer_solver: A valid numerical solver class that will act on an outer loop.
      inner_solver: '' that will act on an inner loop.
      denoise: Bool, that if `True` applies one-step denoising to final samples.
      stack_samples: Bool, that if `True` return all of the sample path or
        just returns the last sample.
      inverse_scaler: The inverse data normalizer function.
    Returns:
      A sampler.
    """
    if inverse_scaler is None:
        inverse_scaler = lambda x: x

    def sampler(rng, x_0=None):
        """
        Args:
          rng: A JAX random state.
          x_0: Initial condition. If `None`, then samples an initial condition from the
              sde's initial condition prior. Note that this initial condition represents
              `x_T sim Normal(O, I)` in reverse-time diffusion.
        Returns:
            Samples and the number of score function (model) evaluations.
        """
        outer_update = partial(shared_update, solver=outer_solver)
        outer_ts = outer_solver.ts

        if inner_solver:
            inner_update = partial(shared_update, solver=inner_solver)
            inner_ts = inner_solver.ts
            num_function_evaluations = jnp.size(outer_ts) * (jnp.size(inner_ts) + 1)

            def inner_step(carry, t):
                rng, x, x_mean, vec_t = carry
                rng, step_rng = random.split(rng)
                x, x_mean = inner_update(step_rng, x, vec_t)
                return (rng, x, x_mean, vec_t), ()

            def outer_step(carry, t):
                rng, x, x_mean = carry
                vec_t = jnp.full(shape[0], t)
                rng, step_rng = random.split(rng)
                x, x_mean = outer_update(step_rng, x, vec_t)
                (rng, x, x_mean, vec_t), _ = scan(
                    inner_step, (step_rng, x, x_mean, vec_t), inner_ts
                )
                if not stack_samples:
                    return (rng, x, x_mean), ()
                else:
                    if denoise:
                        return (rng, x, x_mean), x_mean
                    else:
                        return (rng, x, x_mean), x
        else:
            num_function_evaluations = jnp.size(outer_ts)

            def outer_step(carry, t):
                rng, x, x_mean = carry
                vec_t = jnp.full((shape[0],), t)
                rng, step_rng = random.split(rng)
                x, x_mean = outer_update(step_rng, x, vec_t)
                if not stack_samples:
                    return (rng, x, x_mean), ()
                else:
                    return ((rng, x, x_mean), x_mean) if denoise else ((rng, x, x_mean), x)

        rng, step_rng = random.split(rng)
        if x_0 is None:
            if inner_solver:
                x = inner_solver.prior(step_rng, shape)
            else:
                x = outer_solver.prior(step_rng, shape)
        else:
            assert x_0.shape == shape
            x = x_0

        if not stack_samples:
            (_, x, x_mean), _ = scan(outer_step, (rng, x, x), outer_ts, reverse=True)
            return inverse_scaler(x_mean if denoise else x), num_function_evaluations
        else:
            (_, _, _), xs = scan(outer_step, (rng, x, x), outer_ts, reverse=True)
            return inverse_scaler(xs), num_function_evaluations

    # return jax.pmap(sampler, in_axes=(0), axis_name='batch')
    return sampler


def get_cs_sampler(
    conditioning_method: ConditioningMethodName,
    *,
    sde,
    model,
    sampling_shape,
    inverse_scaler,
    stack_samples,
    y,
    H,
    observation_map,
    **kwargs,
):
    """Create a sampling function

    Args:
        config: A `ml_collections.ConfigDict` object that contains all configuration information.
        sde: A valid SDE class (the forward sde).
        score:
        shape: The shape of array, x. (num_samples,) + x_shape, where x_shape is the shape
            of the object being sampled from, for example, an image may have
            x_shape==(H, W, C), and so shape==(N, H, W, C) where N is the number of samples.
        inverse_scaler: The inverse data normalizer function.
        y: the data
        H: an observation matrix.
        operator_map:
        adjoint_operator_map: TODO generalize like this?

    Returns:
        A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """
    if conditioning_method != ConditioningMethodName.NONE:
        reverse_sde = sde.reverse(model)
        score_func = get_conditioning_method(
            conditioning_method,
            sde=reverse_sde,
            y=y,
            observation_map=observation_map,
            H=H,
            HHT=H @ H.T,  # or exclude for song2023plus
            shape=sampling_shape,
            **kwargs,
        ).guidance_score_func

        sampler = get_sampler(
            sampling_shape,
            EulerMaruyama(
                reverse_sde.guide(score_func)
            ),
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "mpgd":
        # Reproduce MPGD (et al. 2023) paper for VP SDE
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        beta, _ = get_linear_beta_function(
            beta_min=config.model.beta_min, beta_max=config.model.beta_max
        )
        outer_solver = MPGD(
            config.solver.mpgd_scale_hyperparameter,
            y,
            observation_map,
            config.sampling.noise_std,
            sampling_shape[1:],
            model,
            beta,
            ts,
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "dpsddpm":
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        beta, _ = get_linear_beta_function(
            beta_min=config.model.beta_min, beta_max=config.model.beta_max
        )
        score = model
        # Reproduce DPS (Chung et al. 2022) paper for VP SDE
        outer_solver = DPSDDPM(
            config.solver.dps_scale_hyperparameter, y, observation_map, score, beta, ts
        )
        sampler = get_sampler(
            shape=sampling_shape,
            outer_solver=outer_solver,
            inner_solver=None,
            denoise=True,
            stack_samples=stack_samples,
            inverse_scaler=inverse_scaler,
        )
    elif cs_method == "dpsddpmplus":
        score = model
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        beta, _ = get_linear_beta_function(
            beta_min=config.model.beta_min, beta_max=config.model.beta_max
        )
        # Reproduce DPS (Chung et al. 2022) paper for VP SDE
        # https://arxiv.org/pdf/2209.14687.pdf#page=20&zoom=100,144,757
        # https://github.com/DPS2022/diffusion-posterior-sampling/blob/effbde7325b22ce8dc3e2c06c160c021e743a12d/guided_diffusion/condition_methods.py#L86
        # https://github.com/DPS2022/diffusion-posterior-sampling/blob/effbde7325b22ce8dc3e2c06c160c021e743a12d/guided_diffusion/condition_methods.py#L2[…]C47
        outer_solver = DPSDDPMplus(
            config.solver.dps_scale_hyperparameter, y, observation_map, score, beta, ts
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "dpssmld":
        # Reproduce DPS (Chung et al. 2022) paper for VE SDE
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        sigma = get_sigma_function(
            sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max
        )
        score = model
        outer_solver = DPSSMLD(
            config.solver.dps_scale_hyperparameter, y, observation_map, score, sigma, ts
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "dpssmldplus":
        # Reproduce DPS (Chung et al. 2022) paper for VE SDE
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        sigma = get_sigma_function(
            sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max
        )
        score = model
        outer_solver = DPSSMLDplus(
            config.solver.dps_scale_hyperparameter, y, observation_map, score, sigma, ts
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "kpddpm":
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        beta, _ = get_linear_beta_function(
            beta_min=config.model.beta_min, beta_max=config.model.beta_max
        )
        score = model
        outer_solver = KPDDPM(
            y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, beta, ts
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "kpsmld":
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        sigma = get_sigma_function(
            sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max
        )
        score = model
        outer_solver = KPSMLD(
            y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, sigma, ts
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "kpsmlddiag":
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        sigma = get_sigma_function(
            sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max
        )
        score = model
        outer_solver = KPSMLDdiag(
            y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, sigma, ts
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "kpddpmdiag":
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        beta, _ = get_linear_beta_function(
            beta_min=config.model.beta_min, beta_max=config.model.beta_max
        )
        score = model
        outer_solver = KPDDPMdiag(
            y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, beta, ts
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "kpddpmplus":
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        beta, _ = get_linear_beta_function(
            beta_min=config.model.beta_min, beta_max=config.model.beta_max
        )
        score = model
        outer_solver = KPDDPMplus(
            y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, beta, ts
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "kpsmldplus":
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        sigma = get_sigma_function(
            sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max
        )
        score = model
        outer_solver = KPSMLDplus(
            y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, sigma, ts
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "reproducepigdmvp":
        # Reproduce PiGDM (Song et al. 2023) paper for VP SDE
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        beta, _ = get_linear_beta_function(
            beta_min=config.model.beta_min, beta_max=config.model.beta_max
        )
        outer_solver = ReproducePiGDMVP(
            y,
            observation_map,
            config.sampling.noise_std,
            sampling_shape[:1],
            model,
            data_variance=1.0,
            eta=config.solver.eta,
            beta=beta,
            ts=ts,
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "reproducepigdmvpplus":
        # Reproduce PiGDM (Song et al. 2023) paper for VP SDE
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        beta, _ = get_linear_beta_function(
            beta_min=config.model.beta_min, beta_max=config.model.beta_max
        )
        outer_solver = ReproducePiGDMVPplus(
            y,
            observation_map,
            config.sampling.noise_std,
            sampling_shape[1:],
            model,
            data_variance=1.0,
            eta=config.solver.eta,
            beta=beta,
            ts=ts,
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "pigdmvp":
        # Based on PiGDM (Song et al. 2023) paper for VP SDE
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        beta, _ = get_linear_beta_function(
            beta_min=config.model.beta_min, beta_max=config.model.beta_max
        )
        outer_solver = PiGDMVP(
            y,
            observation_map,
            config.sampling.noise_std,
            sampling_shape[:1],
            model,
            data_variance=1.0,
            eta=config.solver.eta,
            beta=beta,
            ts=ts,
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "pigdmvpplus":
        # Based on PiGDM (Song et al. 2023) paper for VP SDE
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        beta, _ = get_linear_beta_function(
            beta_min=config.model.beta_min, beta_max=config.model.beta_max
        )
        outer_solver = PiGDMVPplus(
            y,
            observation_map,
            config.sampling.noise_std,
            sampling_shape[1:],
            model,
            data_variance=1.0,
            eta=config.solver.eta,
            beta=beta,
            ts=ts,
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "pigdmve":
        # Reproduce PiGDM (Song et al. 2023) paper for VE SDE
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        sigma = get_sigma_function(
            sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max
        )
        outer_solver = PiGDMVE(
            y,
            observation_map,
            config.sampling.noise_std,
            sampling_shape[1:],
            model,
            data_variance=1.0,
            eta=config.solver.eta,
            sigma=sigma,
            ts=ts,
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "pigdmveplus":
        # Reproduce PiGDM (Song et al. 2023) paper for VE SDE
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        sigma = get_sigma_function(
            sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max
        )
        outer_solver = PiGDMVEplus(
            y,
            observation_map,
            config.sampling.noise_std,
            sampling_shape[1:],
            model,
            data_variance=1.0,
            eta=config.solver.eta,
            sigma=sigma,
            ts=ts,
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "kgdmvp":
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        beta, _ = get_linear_beta_function(
            beta_min=config.model.beta_min, beta_max=config.model.beta_max
        )
        outer_solver = KGDMVP(
            y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, sigma, ts
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "kgdmve":
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        sigma = get_sigma_function(
            sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max
        )
        outer_solver = KGDMVE(
            y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, sigma, ts
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "kgdmvpplus":
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        beta, _ = get_linear_beta_function(
            beta_min=config.model.beta_min, beta_max=config.model.beta_max
        )
        outer_solver = KGDMVPplus(
            y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, beta, ts
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    elif cs_method == "kgdmveplus":
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        sigma = get_sigma_function(
            sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max
        )
        outer_solver = KGDMVEplus(
            y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, sigma, ts
        )
        sampler = get_sampler(
            sampling_shape,
            outer_solver,
            inverse_scaler=inverse_scaler,
            stack_samples=stack_samples,
            denoise=True,
        )
    else:
        raise ValueError(f"{conditioning_method=} not recognized")

    return sampler


def get_ddim_chain(config, model):
    """
    Args:
        model: DDIM parameterizes the `epsilon(x, t) = -1. * fwd_marginal_std(t) * score(x, t)` function
    """
    if config.solver.outer_solver.lower() == "ddimvp":
        ts, _ = get_times(
            config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        beta, _ = get_linear_beta_function(
            beta_min=config.model.beta_min, beta_max=config.model.beta_max
        )
        return DDIMVP(model, eta=config.solver.eta, beta=beta, ts=ts)
    elif config.solver.outer_solver.lower() == "ddimve":
        ts, _ = get_times(
            config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        sigma = get_sigma_function(
            sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max
        )
        return DDIMVE(model, eta=config.solver.eta, sigma=sigma, ts=ts)
    else:
        raise NotImplementedError(f"DDIM Chain {config.solver.outer_solver} unknown.")


def get_markov_chain(config, score):
    """
    Args:
      score: DDPM/SMLD(NCSN) parameterizes the `score(x, t)` function.
    """
    if config.solver.outer_solver.lower() == "ddpm":
        ts, _ = get_times(
            num_steps=config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        beta, _ = get_linear_beta_function(
            beta_min=config.model.beta_min, beta_max=config.model.beta_max
        )
        return DDPM(score, beta=beta, ts=ts)
    elif config.solver.outer_solver.lower() == "smld":
        ts, _ = get_times(
            config.solver.num_outer_steps, dt=config.solver.dt, t0=config.solver.epsilon
        )
        sigma = get_sigma_function(
            sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max
        )
        return SMLD(score, sigma=sigma, ts=ts)
    else:
        raise NotImplementedError(f"Markov Chain {config.solver.outer_solver} unknown.")


class Solver(ABC):
    """SDE solver abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, ts=None):
        """Construct a Solver. Note that for continuous time we choose to control for numerical
        error by using a beta schedule instead of an adaptive time step schedule, since adaptive
        time steps are equivalent to a beta schedule, and beta schedule hyperparameters have
        been explored extensively in the literature. Therefore, the timesteps must be equally
        spaced by dt.
        Args:
          ts: JAX array of equally spaced, monotonically increasing values t in [t0, t1].
        """
        if ts is None:
            ts, _ = get_times(num_steps=1000)
        self.ts = ts
        self.t1 = ts[-1]
        self.t0 = ts[0]
        self.dt = ts[1] - ts[0]
        self.num_steps = ts.size

    @abstractmethod
    def update(self, rng, x, t):
        """Return the update of the state and any auxilliary values.

        Args:
          rng: A JAX random state.
          x: A JAX array of the state.
          t: JAX array of the time.

        Returns:
          x: A JAX array of the next state.
          x_mean: A JAX array. The next state without random noise. Useful for denoising.
        """
        raise NotImplementedError


class EulerMaruyama(Solver):
    """Euler Maruyama numerical solver of an SDE.
    Functions are designed for a mini-batch of inputs."""

    def __init__(self, sde, ts=None):
        """Constructs an Euler-Maruyama Solver.
        Args:
          sde: A valid SDE class.
        """
        super().__init__(ts)
        self.sde = sde
        self.prior = sde.prior

    def update(self, rng, x, t):
        drift, diffusion = self.sde.sde(x, t)

        noise = random.normal(rng, x.shape)
        f = drift * self.dt
        G = diffusion * jnp.sqrt(self.dt)

        x_mean = x + f
        x = x_mean + batch_mul(G, noise)

        return x, x_mean


class Annealed(Solver):
    """Annealed Langevin numerical solver of an SDE.
    Functions are designed for a mini-batch of inputs.
    Sampler must be `pmap` over "batch" axis as
    suggested by https://arxiv.org/abs/2011.13456 Song
    et al.
    """

    def __init__(self, sde, snr=1e-2, ts=jnp.empty((2, 1))):
        """Constructs an Annealed Langevin Solver.
        Args:
          sde: A valid SDE class.
          snr: A hyperparameter representing a signal-to-noise ratio.
          ts: For a corrector, just need a placeholder JAX array with length
            number of timesteps of the inner solver.
        """
        super().__init__(ts)
        self.sde = sde
        self.snr = snr
        self.prior = sde.prior

    def update(self, rng, x, t):
        alpha = jnp.exp(2 * self.sde.log_mean_coeff(t))

        grad = self.sde.score(x, t)
        grad_norm = jnp.linalg.norm(grad.reshape((grad.shape[0], -1)), axis=-1).mean()
        grad_norm = jax.lax.pmean(grad_norm, axis_name="batch")

        noise = random.normal(rng, x.shape)
        noise_norm = jnp.linalg.norm(noise.reshape((noise.shape[0], -1)), axis=-1).mean()
        noise_norm = jax.lax.pmean(noise_norm, axis_name="batch")

        dt = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + batch_mul(grad, dt)
        x = x_mean + batch_mul(2 * dt, noise)
        return x, x_mean


class Inpainted(Solver):
    """Inpainting constraint for numerical solver of an SDE.
    Functions are designed for a mini-batch of inputs."""

    def __init__(self, sde, mask, y, ts=jnp.empty((1, 1))):
        """Constructs an Annealed Langevin Solver.
        Args:
          sde: A valid SDE class.
          snr: A hyperparameter representing a signal-to-noise ratio.
        """
        super().__init__(ts)
        self.sde = sde
        self.mask = mask
        self.y = y

    def prior(self, rng, shape):
        x = self.sde.prior(rng, shape)
        x = batch_mul_A((1.0 - self.mask), x) + self.y * self.mask
        return x

    def update(self, rng, x, t):
        mean_coeff = self.sde.mean_coeff(t)
        std = jnp.sqrt(self.sde.variance(t))
        masked_data_mean = batch_mul_A(self.y, mean_coeff)
        masked_data = masked_data_mean + batch_mul(random.normal(rng, x.shape), std)
        x = batch_mul_A((1.0 - self.mask), x) + batch_mul_A(self.mask, masked_data)
        x_mean = batch_mul_A((1.0 - self.mask), x) + batch_mul_A(self.mask, masked_data_mean)
        return x, x_mean


class Projected(Solver):
    """Inpainting constraint for numerical solver of an SDE.
    Functions are designed for a mini-batch of inputs."""

    def __init__(self, sde, mask, y, coeff=1.0, ts=jnp.empty((1, 1))):
        """Constructs an Annealed Langevin Solver.
        Args:
          sde: A valid SDE class.
          snr: A hyperparameter representing a signal-to-noise ratio.
        """
        super().__init__(ts)
        self.sde = sde
        self.mask = mask
        self.y = y
        self.coeff = coeff
        self.prior = sde.prior

    def merge_data_with_mask(self, x_space, data, mask, coeff):
        return batch_mul_A(mask * coeff, data) + batch_mul_A((1.0 - mask * coeff), x_space)

    def update(self, rng, x, t):
        mean_coeff = self.sde.mean_coeff(t)
        masked_data_mean = batch_mul_A(self.y, mean_coeff)
        std = jnp.sqrt(self.sde.variance(t))
        z_data = masked_data_mean + batch_mul(std, random.normal(rng, x.shape))
        x = self.merge_data_with_mask(x, z_data, self.mask, self.coeff)
        x_mean = self.merge_data_with_mask(x, masked_data_mean, self.mask, self.coeff)
        return x, x_mean


class DDPM(Solver):
    """DDPM Markov chain using Ancestral sampling."""

    def __init__(self, score, beta=None, ts=None):
        super().__init__(ts)
        if beta is None:
            beta, _ = get_linear_beta_function(beta_min=0.1, beta_max=20.0)
        self.discrete_betas = continuous_to_discrete(vmap(beta)(self.ts.flatten()), self.dt)
        self.score = score
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        self.alphas_cumprod_prev = jnp.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = jnp.sqrt(self.alphas_cumprod_prev)
        self.sqrt_1m_alphas_cumprod_prev = jnp.sqrt(1.0 - self.alphas_cumprod_prev)

    def get_estimate_x_0_vmap(self, observation_map, clip=False, centered=True):
        if clip:
            (a_min, a_max) = (-1.0, 1.0) if centered else (0.0, 1.0)

        def estimate_x_0(x, t, timestep):
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            m = self.sqrt_alphas_cumprod[timestep]
            v = self.sqrt_1m_alphas_cumprod[timestep] ** 2
            s = self.score(x, t)
            x_0 = (x + v * s) / m
            if clip:
                x_0 = jnp.clip(x_0, a_min=a_min, a_max=a_max)
            return observation_map(x_0), (s, x_0)

        return estimate_x_0

    def get_estimate_x_0(self, observation_map, clip=False, centered=True):
        if clip:
            (a_min, a_max) = (-1.0, 1.0) if centered else (0.0, 1.0)
        batch_observation_map = vmap(observation_map)

        def estimate_x_0(x, t, timestep):
            m = self.sqrt_alphas_cumprod[timestep]
            v = self.sqrt_1m_alphas_cumprod[timestep] ** 2
            s = self.score(x, t)
            x_0 = batch_mul(x + batch_mul(v, s), 1.0 / m)
            if clip:
                x_0 = jnp.clip(x_0, a_min=a_min, a_max=a_max)
            return batch_observation_map(x_0), (s, x_0)

        return estimate_x_0

    def prior(self, rng, shape):
        return random.normal(rng, shape)

    def posterior(self, score, x, timestep):
        beta = self.discrete_betas[timestep]
        # As implemented by Song
        # https://github.com/yang-song/score_sde/blob/0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c/sampling.py#L237C5-L237C79
        # x_mean = batch_mul(
        #     (x + batch_mul(beta, score)), 1. / jnp.sqrt(1. - beta))  # DDPM
        # std = jnp.sqrt(beta)

        # # As implemented by DPS2022
        # https://github.com/DPS2022/diffusion-posterior-sampling/blob/effbde7325b22ce8dc3e2c06c160c021e743a12d/guided_diffusion/gaussian_diffusion.py#L373
        m = self.sqrt_alphas_cumprod[timestep]
        v = self.sqrt_1m_alphas_cumprod[timestep] ** 2
        alpha = self.alphas[timestep]
        x_0 = batch_mul((x + batch_mul(v, score)), 1.0 / m)
        m_prev = self.sqrt_alphas_cumprod_prev[timestep]
        v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep] ** 2
        x_mean = batch_mul(jnp.sqrt(alpha) * v_prev / v, x) + batch_mul(m_prev * beta / v, x_0)
        std = jnp.sqrt(beta * v_prev / v)
        return x_mean, std

    def update(self, rng, x, t):
        score = self.score(x, t)
        timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
        x_mean, std = self.posterior(score, x, timestep)
        z = random.normal(rng, x.shape)
        x = x_mean + batch_mul(std, z)
        return x, x_mean


# previous port... for reference
@dataclass
class DDPM_:
    betas: Array
    model_mean_type: MeanProcessorType
    model_var_type: VarianceProcessorType
    clip_denoised: bool = True

    def __post_init__(self):
        assert self.betas.ndim == 1, "betas must be 1-D"
        assert (0 < self.betas).all() and (self.betas <= 1).all(), "betas must be in (0..1]"

        self.num_timesteps = int(self.betas.shape[0])
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = jnp.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = jnp.append(self.alphas_cumprod[1:], 0.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = jnp.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # NOTE: log-calc clipped as posterior variance is 0 at the beginning of the chain.
        self.posterior_log_variance_clipped = jnp.log(
            jnp.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            self.betas * jnp.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * jnp.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

        self.mean_processor = MeanProcessor.from_name(
            self.model_mean_type,
            betas=self.betas,
            clip_denoised=self.clip_denoised,
        )

        self.var_processor = VarianceProcessor.from_name(
            self.model_var_type,
            betas=self.betas,
            posterior_variance=self.posterior_variance,
            posterior_log_variance_clipped=self.posterior_log_variance_clipped,
        )

    @classmethod
    def from_beta_range(
        cls, n: int, beta_min: float = 0.1, beta_max: float = 20.0, **kwargs
    ) -> "DDPM_":
        betas = jnp.array(
            [beta_min / n + i / (n * (n - 1)) * (beta_max - beta_min) for i in range(n)]
        )

        return cls(betas=betas, **kwargs)

    def q_sample(self, x_0: Array, t: Array, key: PRNGKeyArray):
        """Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).
        """
        noise = random.normal(key, x_0.shape)
        coef1 = extract_and_expand(self.sqrt_alphas_cumprod, t, x_0)
        coef2 = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_0)

        return coef1 * x_0 + coef2 * noise

    def q_posterior_mean_variance(self, x_0: Array, x_t: Array, t: Array):
        """Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)"""
        coef1 = extract_and_expand(self.posterior_mean_coef1, t, x_0)
        coef2 = extract_and_expand(self.posterior_mean_coef2, t, x_t)
        posterior_mean = coef1 * x_0 + coef2 * x_t
        posterior_variance = extract_and_expand(self.posterior_variance, t, x_t)
        posterior_log_variance_clipped = extract_and_expand(
            self.posterior_log_variance_clipped, t, x_t
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_sample_loop(self, model, x_0, measurement, measurement_cond_fn, key: PRNGKeyArray):
        def body_fn(idx, val):
            img, key = val
            key, subkey1, subkey2 = random.split(key, 3)

            time = jnp.array([idx] * img.shape[0])
            out = self.p_sample(x=img, t=time, model=model, key=subkey1)
            noisy_measurement = self.q_sample(measurement, t=time, key=subkey2)

            img, _ = measurement_cond_fn(
                x_t=out["sample"],
                measurement=measurement,
                noisy_measurement=noisy_measurement,
                x_prev=img,
                x_0_hat=out["x_0_hat"],
            )
            return img, key

        # Run the loop
        final_img, _ = lax.fori_loop(0, self.num_timesteps, body_fn, (x_0, key))
        return final_img

    def p_sample(self, model, x, t, key):
        out = self.p_mean_variance(model, x, t)
        sample = out["mean"]

        noise = random.normal(key, x.shape)
        if t != 0:  # no noise when t == 0
            sample += jnp.exp(0.5 * out["log_variance"]) * noise

        return {"sample": sample, "x_0_hat": out["x_0_hat"]}

    def p_mean_variance(self, model, x, t):
        model_output = model(x, t)

        # In the case of "learned" variance, model will give twice channels.
        if model_output.shape[1] == 2 * x.shape[1]:
            model_output, model_var_values = jnp.split(model_output, x.shape[1], axis=1)
        else:
            # The name of variable is wrong.
            # This will just provide shape information, and
            # will not be used for calculating something important in variance.
            model_var_values = model_output

        model_mean, x_0_hat = self.mean_processor.get_mean_and_xstart(x, t, model_output)
        model_variance, model_log_variance = self.var_processor.get_variance(model_var_values, t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "x_0_hat": x_0_hat,
        }


class SMLD(Solver):
    """SMLD(NCSN) Markov Chain using Ancestral sampling."""

    def __init__(self, score, sigma=None, ts=None):
        super().__init__(ts)
        if sigma is None:
            sigma = get_sigma_function(sigma_min=0.01, sigma_max=378.0)
        sigmas = vmap(sigma)(self.ts.flatten())
        self.sigma_max = sigmas[-1]
        self.discrete_sigmas = sigmas
        self.discrete_sigmas_prev = jnp.append(0.0, self.discrete_sigmas[:-1])
        self.score = score

    def get_estimate_x_0_vmap(self, observation_map, clip=False, centered=False):
        if clip:
            (a_min, a_max) = (-1.0, 1.0) if centered else (0.0, 1.0)

        def estimate_x_0(x, t, timestep):
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            v = self.discrete_sigmas[timestep] ** 2
            s = self.score(x, t)
            x_0 = x + v * s
            if clip:
                x_0 = jnp.clip(x_0, a_min=a_min, a_max=a_max)
            return observation_map(x_0), (s, x_0)

        return estimate_x_0

    def get_estimate_x_0(self, observation_map, clip=False, centered=False):
        if clip:
            (a_min, a_max) = (-1.0, 1.0) if centered else (0.0, 1.0)
        batch_observation_map = vmap(observation_map)

        def estimate_x_0(x, t, timestep):
            v = self.discrete_sigmas[timestep] ** 2
            s = self.score(x, t)
            x_0 = x + batch_mul(v, s)
            if clip:
                x_0 = jnp.clip(x_0, a_min=a_min, a_max=a_max)
            return batch_observation_map(x_0), (s, x_0)

        return estimate_x_0

    def prior(self, rng, shape):
        return random.normal(rng, shape) * self.sigma_max

    def posterior(self, score, x, timestep):
        sigma = self.discrete_sigmas[timestep]
        sigma_prev = self.discrete_sigmas_prev[timestep]

        # As implemented by Song https://github.com/yang-song/score_sde/blob/0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c/sampling.py#L220
        # x_mean = x + batch_mul(score, sigma**2 - sigma_prev**2)
        # std = jnp.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))

        # From posterior in Appendix F https://arxiv.org/pdf/2011.13456.pdf
        x_0 = x + batch_mul(sigma**2, score)
        x_mean = batch_mul(sigma_prev**2 / sigma**2, x) + batch_mul(
            1 - sigma_prev**2 / sigma**2, x_0
        )
        std = jnp.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
        return x_mean, std

    def update(self, rng, x, t):
        timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
        score = self.score(x, t)
        x_mean, std = self.posterior(score, x, timestep)
        z = random.normal(rng, x.shape)
        x = x_mean + batch_mul(std, z)
        return x, x_mean


class DDIMVP(Solver):
    """DDIM Markov chain. For the DDPM Markov Chain or VP SDE."""

    def __init__(self, model, eta=1.0, beta=None, ts=None):
        """
        Args:
            model: DDIM parameterizes the `epsilon(x, t) = -1. * fwd_marginal_std(t) * score(x, t)` function.
            eta: the hyperparameter for DDIM, a value of `eta=0.0` is deterministic 'probability ODE' solver, `eta=1.0` is DDPMVP.
        """
        super().__init__(ts)
        if beta is None:
            beta, _ = get_linear_beta_function(beta_min=0.1, beta_max=20.0)
        self.discrete_betas = continuous_to_discrete(vmap(beta)(self.ts.flatten()), self.dt)
        self.eta = eta
        self.model = model
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        self.alphas_cumprod_prev = jnp.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = jnp.sqrt(self.alphas_cumprod_prev)
        self.sqrt_1m_alphas_cumprod_prev = jnp.sqrt(1.0 - self.alphas_cumprod_prev)

    def get_estimate_x_0_vmap(self, observation_map, clip=False, centered=True):
        if clip:
            (a_min, a_max) = (-1.0, 1.0) if centered else (0.0, 1.0)

        def estimate_x_0(x, t, timestep):
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            m = self.sqrt_alphas_cumprod[timestep]
            sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
            epsilon = self.model(x, t)
            x_0 = (x - sqrt_1m_alpha * epsilon) / m
            if clip:
                x_0 = jnp.clip(x_0, a_min=a_min, a_max=a_max)
            return observation_map(x_0), (epsilon, x_0)

        return estimate_x_0

    def get_estimate_x_0(self, observation_map, clip=False, centered=True):
        if clip:
            (a_min, a_max) = (-1.0, 1.0) if centered else (0.0, 1.0)
        batch_observation_map = vmap(observation_map)

        def estimate_x_0(x, t, timestep):
            m = self.sqrt_alphas_cumprod[timestep]
            sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
            epsilon = self.model(x, t)
            x_0 = batch_mul(x - batch_mul(sqrt_1m_alpha, epsilon), 1.0 / m)
            if clip:
                x_0 = jnp.clip(x_0, a_min=a_min, a_max=a_max)
            return batch_observation_map(x_0), (epsilon, x_0)

        return estimate_x_0

    def prior(self, rng, shape):
        return random.normal(rng, shape)

    def posterior(self, x, t):
        # # As implemented by DPS2022
        # https://github.com/DPS2022/diffusion-posterior-sampling/blob/effbde7325b22ce8dc3e2c06c160c021e743a12d/guided_diffusion/gaussian_diffusion.py#L373
        # and as written in https://arxiv.org/pdf/2010.02502.pdf
        epsilon = self.model(x, t)
        timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
        m = self.sqrt_alphas_cumprod[timestep]
        sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
        v = sqrt_1m_alpha**2
        alpha_cumprod = self.alphas_cumprod[timestep]
        alpha_cumprod_prev = self.alphas_cumprod_prev[timestep]
        m_prev = self.sqrt_alphas_cumprod_prev[timestep]
        v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep] ** 2
        x_0 = batch_mul((x - batch_mul(sqrt_1m_alpha, epsilon)), 1.0 / m)
        coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha_cumprod / alpha_cumprod_prev))
        coeff2 = jnp.sqrt(v_prev - coeff1**2)
        x_mean = batch_mul(m_prev, x_0) + batch_mul(coeff2, epsilon)
        std = coeff1
        return x_mean, std

    def update(self, rng, x, t):
        x_mean, std = self.posterior(x, t)
        z = random.normal(rng, x.shape)
        x = x_mean + batch_mul(std, z)
        return x, x_mean


class DDIMVE(Solver):
    """DDIM Markov chain. For the SMLD Markov Chain or VE SDE.
    Args:
        model: DDIM parameterizes the `epsilon(x, t) = -1. * fwd_marginal_std(t) * score(x, t)` function.
        eta: the hyperparameter for DDIM, a value of `eta=0.0` is deterministic 'probability ODE' solver, `eta=1.0` is DDPMVE.
    """

    def __init__(self, model, eta=1.0, sigma=None, ts=None):
        super().__init__(ts)
        if sigma is None:
            sigma = get_sigma_function(sigma_min=0.01, sigma_max=378.0)
        sigmas = vmap(sigma)(self.ts.flatten())
        self.sigma_max = sigmas[-1]
        self.discrete_sigmas = sigmas
        self.discrete_sigmas_prev = jnp.append(0.0, self.discrete_sigmas[:-1])
        self.eta = eta
        self.model = model

    def get_estimate_x_0_vmap(self, observation_map, clip=False, centered=False):
        if clip:
            a_min, a_max = (-1.0, 1.0) if centered else (0.0, 1.0)

        def estimate_x_0(x, t, timestep):
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            std = self.discrete_sigmas[timestep]
            epsilon = self.model(x, t)
            x_0 = x - std * epsilon
            if clip:
                x_0 = jnp.clip(x_0, a_min=a_min, a_max=a_max)
            return observation_map(x_0), (epsilon, x_0)

        return estimate_x_0

    def get_estimate_x_0(self, observation_map, clip=False, centered=False):
        if clip:
            a_min, a_max = (-1.0, 1.0) if centered else (0.0, 1.0)
        batch_observation_map = vmap(observation_map)

        def estimate_x_0(x, t, timestep):
            std = self.discrete_sigmas[timestep]
            epsilon = self.model(x, t)
            x_0 = x - batch_mul(std, epsilon)
            if clip:
                x_0 = jnp.clip(x_0, a_min=a_min, a_max=a_max)
            return batch_observation_map(x_0), (epsilon, x_0)

        return estimate_x_0

    def prior(self, rng, shape):
        return random.normal(rng, shape) * self.sigma_max

    def posterior(self, x, t):
        timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
        epsilon = self.model(x, t)
        sigma = self.discrete_sigmas[timestep]
        sigma_prev = self.discrete_sigmas_prev[timestep]
        coeff1 = self.eta * jnp.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
        coeff2 = jnp.sqrt(sigma_prev**2 - coeff1**2)

        # Eq.(18) Appendix A.4 https://openreview.net/pdf/210093330709030207aa90dbfe2a1f525ac5fb7d.pdf
        x_0 = x - batch_mul(sigma, epsilon)
        x_mean = x_0 + batch_mul(coeff2, epsilon)

        # Eq.(19) Appendix A.4 https://openreview.net/pdf/210093330709030207aa90dbfe2a1f525ac5fb7d.pdf
        # score = - batch_mul(1. / sigma, epsilon)
        # x_mean = x + batch_mul(sigma * (sigma - coeff2), score)

        std = coeff1
        return x_mean, std

    def update(self, rng, x, t):
        x_mean, std = self.posterior(x, t)
        z = random.normal(rng, x.shape)
        x = x_mean + batch_mul(std, z)
        return x, x_mean

class MPGD(DDIMVP):
    """
    TODO: This method requires a pretrained autoencoder.
    MPGD He, Murata et al. https://arxiv.org/pdf/2311.16424.pdf"""

    def __init__(
        self, scale, y, observation_map, noise_std, shape, model, eta=1.0, beta=None, ts=None
    ):
        super().__init__(model, eta, beta, ts)
        self.observation_map = observation_map
        self.batch_observation_map = vmap(observation_map)
        self.likelihood_score = self.get_likelihood_score(self.batch_observation_map)
        self.likelihood_score_vmap = self.get_likelihood_score_vmap(self.observation_map)
        self.scale = scale
        self.y = y
        self.noise_std = noise_std
        self.num_y = y.shape[1]

    def get_likelihood_score_vmap(self, observation_map):
        def l2_norm(x_0, y):
            x_0 = x_0
            norm = jnp.linalg.norm(y - observation_map(x_0))
            return norm**2 / self.noise_std

        grad_l2_norm = grad(l2_norm)
        return vmap(grad_l2_norm)

    def get_likelihood_score(self, batch_observation_map):
        batch_norm = vmap(jnp.linalg.norm)

        def l2_norm(x_0, y):
            norm = batch_norm(y - batch_observation_map(x_0))
            squared_norm = jnp.sum(norm**2)
            return squared_norm / self.noise_std

        grad_l2_norm = grad(l2_norm)
        return grad_l2_norm

    def posterior(self, x, t):
        epsilon = self.model(x, t)
        timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
        m = self.sqrt_alphas_cumprod[timestep]
        sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
        v = sqrt_1m_alpha**2
        alpha = m**2
        m_prev = self.sqrt_alphas_cumprod_prev[timestep]
        v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep] ** 2
        alpha_prev = m_prev**2
        x_0 = batch_mul((x - batch_mul(sqrt_1m_alpha, epsilon)), 1.0 / m)
        # TODO: manifold projection step here which requires autoencoder to project x_0 such that likelihood score is projected onto tangent space of autoencoder
        # ls = self.likelihood_score(x_0, self.y)
        ls = self.likelihood_score_vmap(x_0, self.y)
        x_0 = x_0 - self.scale * ls
        coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha / alpha_prev))
        coeff2 = jnp.sqrt(v_prev - coeff1**2)
        x_mean = batch_mul(m_prev, x_0) + batch_mul(coeff2, epsilon)
        std = coeff1
        return x_mean, std


class KGDMVP(DDIMVP):
    """Kalman Guided Diffusion Model, Markov chain using the DDIM Markov Chain or VP SDE."""

    def __init__(self, y, observation_map, noise_std, shape, model, eta=1.0, beta=None, ts=None):
        super().__init__(model, eta, beta, ts)
        self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map)
        self.batch_analysis_vmap = vmap(self.analysis)
        self.y = y
        self.noise_std = noise_std
        self.num_y = y.shape[1]
        self.observation_map = observation_map
        self.batch_observation_map = vmap(observation_map)
        self.axes_vmap = tuple(range(len(shape) + 1)[1:]) + (0,)

    def analysis(self, y, x, t, timestep, ratio):
        h_x_0, (epsilon, _) = self.estimate_h_x_0_vmap(
            x, t, timestep
        )  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0_vmap(_x, t, timestep)[0])(x)
        H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
        C_yy = ratio * H_grad_H_x_0 + self.noise_std**2 * jnp.eye(self.num_y)
        f = jnp.linalg.solve(C_yy, y - h_x_0)
        ls = grad_H_x_0.transpose(self.axes_vmap) @ f
        return epsilon.squeeze(axis=0), ls

    def posterior(self, x, t):
        timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
        m = self.sqrt_alphas_cumprod[timestep]
        sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
        v = sqrt_1m_alpha**2
        ratio = v / m
        alpha = m**2
        epsilon, ls = self.batch_analysis_vmap(self.y, x, t, timestep, ratio)
        m_prev = self.sqrt_alphas_cumprod_prev[timestep]
        v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep] ** 2
        alpha_prev = m_prev**2
        coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha / alpha_prev))
        coeff2 = jnp.sqrt(v_prev - coeff1**2)
        posterior_score = -batch_mul(1.0 / sqrt_1m_alpha, epsilon) + ls
        x_mean = batch_mul(m_prev / m, x) + batch_mul(
            sqrt_1m_alpha * (sqrt_1m_alpha * m_prev / m - coeff2), posterior_score
        )
        std = coeff1
        return x_mean, std


class KGDMVPplus(KGDMVP):
    """KGDMVP with a mask."""

    def analysis(self, y, x, t, timestep, ratio):
        h_x_0, vjp_h_x_0, (epsilon, _) = vjp(
            lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
        )
        C_yy = (
            ratio * self.observation_map(vjp_h_x_0(self.observation_map(jnp.ones_like(x)))[0])
            + self.noise_std**2
        )
        ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
        return epsilon.squeeze(axis=0), ls


class KGDMVE(DDIMVE):
    def __init__(self, y, observation_map, noise_std, shape, model, eta=1.0, sigma=None, ts=None):
        super().__init__(model, eta, sigma, ts)
        self.eta = eta
        self.model = model
        self.discrete_sigmas = jnp.exp(
            jnp.linspace(jnp.log(self.sigma_min), jnp.log(self.sigma_max), self.num_steps)
        )
        self.y = y
        self.noise_std = noise_std
        self.num_y = y.shape[1]
        self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map)
        self.batch_analysis_vmap = vmap(self.analysis)
        self.observation_map = observation_map
        self.batch_observation_map = vmap(observation_map)
        self.axes_vmap = tuple(range(len(shape) + 1)[1:]) + (0,)

    def analysis(self, y, x, t, timestep, ratio):
        h_x_0, (epsilon, _) = self.estimate_h_x_0_vmap(
            x, t, timestep
        )  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0_vmap(_x, t, timestep)[0])(x)
        H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
        C_yy = ratio * H_grad_H_x_0 + self.noise_std**2 * jnp.eye(self.num_y)
        f = jnp.linalg.solve(C_yy, y - h_x_0)
        ls = grad_H_x_0.transpose(self.axes_vmap) @ f
        return epsilon.squeeze(axis=0), ls

    def posterior(self, x, t):
        timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
        sigma = self.discrete_sigmas[timestep]
        sigma_prev = self.discrete_sigmas_prev[timestep]
        epsilon, ls = self.batch_analysis_vmap(self.y, x, t, timestep, sigma**2)
        coeff1 = self.eta * jnp.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
        coeff2 = jnp.sqrt(sigma_prev**2 - coeff1**2)
        std = coeff1
        posterior_score = -batch_mul(1.0 / sigma, epsilon) + ls
        x_mean = x + batch_mul(sigma * (sigma - coeff2), posterior_score)
        return x_mean, std


class KGDMVEplus(KGDMVE):
    """KGDMVE with a mask."""

    def analysis(self, y, x, t, timestep, ratio):
        h_x_0, vjp_h_x_0, (epsilon, _) = vjp(
            lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
        )
        C_yy = (
            ratio * self.observation_map(vjp_h_x_0(self.observation_map(jnp.ones_like(x)))[0])
            + self.noise_std**2
        )
        ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
        return epsilon.squeeze(axis=0), ls


class PiGDMVP(DDIMVP):
    """PiGDM Song et al. 2023. Markov chain using the DDIM Markov Chain or VP SDE."""

    def __init__(
        self,
        y,
        observation_map,
        noise_std,
        shape,
        model,
        data_variance=1.0,
        eta=1.0,
        beta=None,
        ts=None,
    ):
        super().__init__(model, eta, beta, ts)
        # This method requires clipping in order to remain (robustly, over all samples) numerically stable
        self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(
            observation_map, clip=True, centered=True
        )
        self.batch_analysis_vmap = vmap(self.analysis)
        self.y = y
        self.noise_std = noise_std
        self.data_variance = data_variance
        self.num_y = y.shape[1]
        self.observation_map = observation_map

    def analysis(self, y, x, t, timestep, v, alpha):
        h_x_0, vjp_h_x_0, (epsilon, _) = vjp(
            lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
        )
        # Value suggested for VPSDE in original PiGDM paper
        r = v * self.data_variance / (v + alpha * self.data_variance)
        C_yy = 1.0 + self.noise_std**2 / r
        ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
        return epsilon.squeeze(axis=0), ls

    def posterior(self, x, t):
        timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
        m = self.sqrt_alphas_cumprod[timestep]
        sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
        v = sqrt_1m_alpha**2
        alpha = self.alphas_cumprod[timestep]
        epsilon, ls = self.batch_analysis_vmap(self.y, x, t, timestep, v, alpha)
        m_prev = self.sqrt_alphas_cumprod_prev[timestep]
        v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep] ** 2
        alpha_prev = self.alphas_cumprod_prev[timestep]
        coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha / alpha_prev))
        coeff2 = jnp.sqrt(v_prev - coeff1**2)
        # TODO: slightly different to Algorithm 1
        posterior_score = -batch_mul(1.0 / sqrt_1m_alpha, epsilon) + ls
        x_mean = batch_mul(m_prev / m, x) + batch_mul(
            sqrt_1m_alpha * (sqrt_1m_alpha * m_prev / m - coeff2), posterior_score
        )
        std = coeff1
        return x_mean, std


class PiGDMVPplus(PiGDMVP):
    """PiGDMVP with a mask."""

    def analysis(self, y, x, t, timestep, v, alpha):
        h_x_0, vjp_estimate_h_x_0, (epsilon, _) = vjp(
            lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
        )
        # Value suggested for VPSDE in original PiGDM paper
        r = v * self.data_variance / (v + alpha * self.data_variance)
        C_yy = 1.0 + self.noise_std**2 / r
        ls = vjp_estimate_h_x_0((y - h_x_0) / C_yy)[0]
        return epsilon.squeeze(axis=0), ls


class ReproducePiGDMVP(DDIMVP):
    """
    NOTE: We found this method to be unstable on CIFAR10 dataset, even with
      thresholding (clip=True) is used at each step of estimating x_0, and for each weighting
      schedule that we tried.
    PiGDM Song et al. 2023. Markov chain using the DDIM Markov Chain or VP SDE."""

    def __init__(
        self,
        y,
        observation_map,
        noise_std,
        shape,
        model,
        data_variance=1.0,
        eta=1.0,
        beta=None,
        ts=None,
    ):
        super().__init__(model, eta, beta, ts)
        self.data_variance = data_variance
        self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(
            observation_map, clip=True, centered=True
        )
        self.batch_analysis_vmap = vmap(self.analysis)
        self.y = y
        self.noise_std = noise_std
        self.num_y = y.shape[1]

    def analysis(self, y, x, t, timestep, v, alpha):
        h_x_0, vjp_estimate_h_x_0, (epsilon, x_0) = vjp(
            lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
        )
        # Value suggested for VPSDE in original PiGDM paper:
        r = v * self.data_variance / (v + self.data_variance)
        # What it should really be set to, following the authors' mathematical reasoning:
        # r = v * self.data_variance  / (v + alpha * self.data_variance)
        C_yy = 1.0 + self.noise_std**2 / r
        ls = vjp_estimate_h_x_0((y - h_x_0) / C_yy)[0]
        return x_0.squeeze(axis=0), ls, epsilon.squeeze(axis=0)

    def posterior(self, x, t):
        timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
        m = self.sqrt_alphas_cumprod[timestep]
        sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
        v = sqrt_1m_alpha**2
        alpha = self.alphas_cumprod[timestep]
        x_0, ls, epsilon = self.batch_analysis_vmap(self.y, x, t, timestep, v, alpha)
        m_prev = self.sqrt_alphas_cumprod_prev[timestep]
        v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep] ** 2
        alpha_prev = self.alphas_cumprod_prev[timestep]
        coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha / alpha_prev))
        coeff2 = jnp.sqrt(v_prev - coeff1**2)
        x_mean = batch_mul(m_prev, x_0) + batch_mul(coeff2, epsilon) + batch_mul(m, ls)
        std = coeff1
        return x_mean, std


class ReproducePiGDMVPplus(ReproducePiGDMVP):
    """
    NOTE: We found this method to be unstable on CIFAR10 dataset, even with
      thresholding (clip=True) is used at each step of estimating x_0, and for each weighting
      schedule that we tried.
    PiGDM with a mask. Song et al. 2023. Markov chain using the DDIM Markov Chain or VP SDE."""

    def analysis(self, y, x, t, timestep, v, alpha):
        h_x_0, vjp_estimate_h_x_0, (epsilon, x_0) = vjp(
            lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
        )
        # Value suggested for VPSDE in original PiGDM paper:
        r = v * self.data_variance / (v + self.data_variance)
        # What it should really be set to, following the authors' mathematical reasoning:
        # r = v * self.data_variance  / (v + alpha * self.data_variance)
        C_yy = 1.0 + self.noise_std**2 / r
        ls = vjp_estimate_h_x_0((y - h_x_0) / C_yy)[0]
        return x_0.squeeze(axis=0), ls, epsilon.squeeze(axis=0)


class PiGDMVE(DDIMVE):
    """PiGDMVE for the SMLD Markov Chain or VE SDE."""

    def __init__(
        self,
        y,
        observation_map,
        noise_std,
        shape,
        model,
        data_variance=1.0,
        eta=1.0,
        sigma=None,
        ts=None,
    ):
        super().__init__(model, eta, sigma, ts)
        self.y = y
        self.data_variance = data_variance
        self.noise_std = noise_std
        self.num_y = y.shape[1]
        # This method requires clipping in order to remain (robustly, over all samples) numerically stable
        self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(
            observation_map, clip=True, centered=False
        )
        self.batch_analysis_vmap = vmap(self.analysis)

    def analysis(self, y, x, t, timestep, v):
        h_x_0, vjp_h_x_0, (epsilon, x_0) = vjp(
            lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
        )
        r = v * self.data_variance / (v + self.data_variance)
        C_yy = 1.0 + self.noise_std**2 / r
        ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
        return x_0.squeeze(axis=0), ls, epsilon.squeeze(axis=0)

    def posterior(self, x, t):
        timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
        sigma = self.discrete_sigmas[timestep]
        sigma_prev = self.discrete_sigmas_prev[timestep]
        x_0, ls, epsilon = self.batch_analysis_vmap(self.y, x, t, timestep, sigma**2)
        coeff1 = self.eta * jnp.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
        coeff2 = jnp.sqrt(sigma_prev**2 - coeff1**2)
        x_mean = x_0 + batch_mul(coeff2, epsilon) + ls
        std = coeff1
        return x_mean, std


class PiGDMVEplus(PiGDMVE):
    """KGDMVE with a mask."""

    def analysis(self, y, x, t, timestep, v):
        h_x_0, vjp_h_x_0, (epsilon, x_0) = vjp(
            lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
        )
        # Value suggested for VPSDE in original PiGDM paper
        r = v * self.data_variance / (v + self.data_variance)
        C_yy = 1.0 + self.noise_std**2 / r
        ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
        return x_0.squeeze(axis=0), ls, epsilon.squeeze(axis=0)


class DPSSMLD(SMLD):
    """DPS for SMLD ancestral sampling.
    NOTE: This method requires static thresholding (clip=True) in order to remain
      (robustly, over all samples) numerically stable"""

    def __init__(self, scale, y, observation_map, score, sigma=None, ts=None):
        super().__init__(score, sigma, ts)
        self.y = y
        self.scale = scale
        self.likelihood_score = self.get_likelihood_score(
            self.get_estimate_x_0(observation_map, clip=True, centered=False)
        )
        self.likelihood_score_vmap = self.get_likelihood_score_vmap(
            self.get_estimate_x_0_vmap(observation_map, clip=True, centered=False)
        )

    def get_likelihood_score_vmap(self, estimate_h_x_0_vmap):
        def l2_norm(x, t, timestep, y):
            h_x_0, (s, _) = estimate_h_x_0_vmap(x, t, timestep)
            norm = jnp.linalg.norm(y - h_x_0)
            return norm, s.squeeze(axis=0)

        grad_l2_norm = grad(l2_norm, has_aux=True)
        return vmap(grad_l2_norm)

    def get_likelihood_score(self, estimate_h_x_0):
        batch_norm = vmap(jnp.linalg.norm)

        def l2_norm(x, t, timestep, y):
            h_x_0, (s, _) = estimate_h_x_0(x, t, timestep)
            norm = batch_norm(y - h_x_0)
            norm = jnp.sum(norm)
            return norm, s

        grad_l2_norm = grad(l2_norm, has_aux=True)
        return grad_l2_norm

    def update(self, rng, x, t):
        """Return the update of the state and any auxilliary variables."""
        timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
        # ls, score = self.likelihood_score(x, t, timestep, self.y)
        ls, score = self.likelihood_score_vmap(x, t, timestep, self.y)
        x_mean, std = self.posterior(score, x, timestep)

        # play around with dps method for the best weighting schedule...
        # sigma = self.discrete_sigmas[timestep]
        # sigma_prev = self.discrete_sigmas_prev[timestep]
        # x_mean = x_mean - batch_mul(1 - sigma_prev**2 / sigma**2, self.scale * ls)
        # x_mean = x_mean - batch_mul(sigma**2, self.scale * ls)
        # Since DPS was empirically derived for VP SDE, the scaling in their paper will not work for VE SDE
        x_mean = x_mean - self.scale * ls  # Not the correct scaling for VE
        z = random.normal(rng, x.shape)
        x = x_mean + batch_mul(std, z)
        return x, x_mean


DPSSMLDplus = DPSSMLD


class DPSDDPM(DDPM):
    """DPS for DDPM ancestral sampling.
    NOTE: This method requires static thresholding (clip=True) in order to remain
      (robustly, over all samples) numerically stable"""

    def __init__(self, scale, y, observation_map, score, beta=None, ts=None):
        super().__init__(score, beta, ts)
        self.y = y
        self.scale = scale
        self.likelihood_score = self.get_likelihood_score(
            self.get_estimate_x_0(observation_map, clip=True, centered=True)
        )
        self.likelihood_score_vmap = self.get_likelihood_score_vmap(
            self.get_estimate_x_0_vmap(observation_map, clip=True, centered=True)
        )

    def get_likelihood_score_vmap(self, estimate_h_x_0_vmap):
        def l2_norm(x, t, timestep, y):
            h_x_0, (s, _) = estimate_h_x_0_vmap(x, t, timestep)
            norm = jnp.linalg.norm(y - h_x_0)
            return norm, s.squeeze(axis=0)

        grad_l2_norm = grad(l2_norm, has_aux=True)
        return vmap(grad_l2_norm)

    def get_likelihood_score(self, estimate_h_x_0):
        batch_norm = vmap(jnp.linalg.norm)

        def l2_norm(x, t, timestep, y):
            h_x_0, (s, _) = estimate_h_x_0(x, t, timestep)
            norm = batch_norm(y - h_x_0)
            norm = jnp.sum(norm)
            return norm, s

        grad_l2_norm = grad(l2_norm, has_aux=True)
        return grad_l2_norm

    def update(self, rng, x, t):
        """Return the update of the state and any auxilliary variables."""
        timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
        # ls, score = self.likelihood_score(x, t, timestep, self.y)
        ls, score = self.likelihood_score_vmap(x, t, timestep, self.y)
        x_mean, std = self.posterior(score, x, timestep)
        x_mean -= self.scale * ls  # DPS
        z = random.normal(rng, x.shape)
        x = x_mean + batch_mul(std, z)
        return x, x_mean


DPSDDPMplus = DPSDDPM


class KPDDPM(DDPM):
    """Kalman posterior for DDPM Ancestral sampling."""

    def __init__(self, y, observation_map, noise_std, shape, score, beta, ts):
        super().__init__(score, beta, ts)
        self.y = y
        self.noise_std = noise_std
        # NOTE: Special case when num_y==1 can be handled correctly by defining observation_map output shape (1,)
        self.num_y = y.shape[1]
        self.shape = shape
        self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map)
        self.batch_analysis_vmap = vmap(self.analysis)
        self.observation_map = observation_map
        self.batch_observation_map = vmap(observation_map)
        self.axes_vmap = tuple(range(len(shape) + 1)[1:]) + (0,)

    def analysis(self, y, x, t, timestep, ratio):
        h_x_0, (_, x_0) = self.estimate_h_x_0_vmap(
            x, t, timestep
        )  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0_vmap(_x, t, timestep)[0])(x)
        H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
        C_yy = H_grad_H_x_0 + self.noise_std**2 / ratio * jnp.eye(self.num_y)
        f = jnp.linalg.solve(C_yy, y - h_x_0)
        ls = grad_H_x_0.transpose(self.axes_vmap) @ f
        return x_0.squeeze(axis=0) + ls

    def posterior(self, x, t):
        timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
        beta = self.discrete_betas[timestep]
        m = self.sqrt_alphas_cumprod[timestep]
        v = self.sqrt_1m_alphas_cumprod[timestep] ** 2
        ratio = v / m
        x_0 = self.batch_analysis_vmap(self.y, x, t, timestep, ratio)
        alpha = self.alphas[timestep]
        m_prev = self.sqrt_alphas_cumprod_prev[timestep]
        v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep] ** 2
        x_mean = batch_mul(jnp.sqrt(alpha) * v_prev / v, x) + batch_mul(m_prev * beta / v, x_0)
        std = jnp.sqrt(beta * v_prev / v)
        return x_mean, std

    def update(self, rng, x, t):
        """Return the update of the state and any auxilliary variables."""
        x_mean, std = self.posterior(x, t)
        z = random.normal(rng, x.shape)
        x = x_mean + batch_mul(std, z)
        return x, x_mean


class KPDDPMplus(KPDDPM):
    """Kalman posterior for DDPM Ancestral sampling."""

    def analysis(self, y, x, t, timestep, ratio):
        h_x_0, vjp_estimate_h_x_0, (_, x_0) = vjp(
            lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
        )
        C_yy = (
            self.observation_map(vjp_estimate_h_x_0(self.observation_map(jnp.ones_like(x)))[0])
            + self.noise_std**2 / ratio
        )
        ls = vjp_estimate_h_x_0((y - h_x_0) / C_yy)[0]
        return x_0.squeeze(axis=0) + ls


class KPSMLD(SMLD):
    """Kalman posterior for SMLD Ancestral sampling."""

    def __init__(self, y, observation_map, noise_std, shape, score, sigma=None, ts=None):
        super().__init__(score, sigma, ts)
        self.y = y
        self.noise_std = noise_std
        self.num_y = y.shape[1]
        self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(
            observation_map, clip=True, centered=False
        )
        self.batch_analysis_vmap = vmap(self.analysis)
        self.observation_map = observation_map
        self.batch_observation_map = vmap(observation_map)
        self.axes_vmap = tuple(range(len(shape) + 1)[1:]) + (0,)

    def analysis(self, y, x, t, timestep, ratio):
        h_x_0, (_, x_0) = self.estimate_h_x_0_vmap(
            x, t, timestep
        )  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0_vmap(_x, t, timestep)[0])(x)
        H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
        C_yy = H_grad_H_x_0 + self.noise_std**2 / ratio * jnp.eye(self.num_y)
        f = jnp.linalg.solve(C_yy, y - h_x_0)
        ls = grad_H_x_0.transpose(self.axes_vmap) @ f
        return x_0.squeeze(axis=0) + ls

    def posterior(self, x, t):
        timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
        sigma = self.discrete_sigmas[timestep]
        sigma_prev = self.discrete_sigmas_prev[timestep]
        x_0 = self.batch_analysis_vmap(self.y, x, t, timestep, sigma**2)
        x_mean = batch_mul(sigma_prev**2 / sigma**2, x) + batch_mul(
            1 - sigma_prev**2 / sigma**2, x_0
        )
        std = jnp.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
        return x_mean, std, x_0

    def update(self, rng, x, t):
        """Return the update of the state and any auxilliary variables."""
        x_mean, std, x_0 = self.posterior(x, t)
        z = random.normal(rng, x.shape)
        x = x_mean + batch_mul(std, z)
        return x, x_0


class KPSMLDplus(KPSMLD):
    """Kalman posterior for SMLD Ancestral sampling."""

    def analysis(self, y, x, t, timestep, ratio):
        h_x_0, vjp_h_x_0, (_, x_0) = vjp(
            lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
        )
        C_yy = (
            self.observation_map(vjp_h_x_0(self.observation_map(jnp.ones_like(x)))[0])
            + self.noise_std**2 / ratio
        )
        ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
        return x_0.squeeze(axis=0) + ls


class KPSMLDdiag(KPSMLD):
    """Kalman posterior for SMLD Ancestral sampling."""
    def analysis(self, y, x, t, timestep, ratio):
        h_x_0, vjp_h_x_0, (_, x_0) = vjp(
            lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
        )

        # There is no natural way to do this with JAX's transforms:
        # you cannot map the input, because in general each diagonal entry of the jacobian depends on all inputs.
        # This seems like the best method, but it is too slow for numerical evaluation, and batch size cannot be large (max size tested was one)
        vec_vjp_h_x_0 = vmap(vjp_h_x_0)
        diag = jnp.diag(self.batch_observation_map(vec_vjp_h_x_0(jnp.eye(y.shape[0]))[0]))
        C_yy = diag + self.noise_std**2 / ratio
        ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
        return x_0.squeeze(axis=0) + ls


class KPDDPMdiag(KPDDPM):
    """Kalman posterior for DDPM Ancestral sampling."""

    def analysis(self, y, x, t, timestep, ratio):
        h_x_0, vjp_h_x_0, (_, x_0) = vjp(
            lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
        )
        vec_vjp_h_x_0 = vmap(vjp_h_x_0)
        diag = jnp.diag(self.batch_observation_map(vec_vjp_h_x_0(jnp.eye(y.shape[0]))[0]))
        C_yy = diag + self.noise_std**2 / ratio
        ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
        return x_0.squeeze(axis=0) + ls
