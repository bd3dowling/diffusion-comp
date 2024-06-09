"""Solver classes, including Markov chains."""
import abc

import jax
import jax.numpy as jnp
import jax.random as random
from jax import grad, jacrev, random, vjp, vmap

from diffusionlib.util.misc import (
    batch_mul,
    batch_mul_A,
    continuous_to_discrete,
    get_linear_beta_function,
    get_sigma_function,
    get_times,
    get_timestep,
)


class Solver(abc.ABC):
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
        # print("t \in [{}, {}] with step size dt={} and num_steps={}".format(
        #   self.t0, self.t1, self.dt, self.num_steps))
        dts = jnp.diff(ts)
        if not jnp.all(dts > 0.0):
            raise ValueError(f"ts must be monotonically increasing, got ts={ts}")
        if not jnp.all(dts == self.dt):
            raise ValueError(
                f"stepsize dt must be constant; ts must be equally \
      spaced, got diff(ts)={dts}"
            )

    @abc.abstractmethod
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
        f = drift * self.dt
        G = diffusion * jnp.sqrt(self.dt)
        noise = random.normal(rng, x.shape)
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
        grad = self.sde.score(x, t)
        grad_norm = jnp.linalg.norm(grad.reshape((grad.shape[0], -1)), axis=-1).mean()
        grad_norm = jax.lax.pmean(grad_norm, axis_name="batch")
        noise = random.normal(rng, x.shape)
        noise_norm = jnp.linalg.norm(noise.reshape((noise.shape[0], -1)), axis=-1).mean()
        noise_norm = jax.lax.pmean(noise_norm, axis_name="batch")
        alpha = jnp.exp(2 * self.sde.log_mean_coeff(t))
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

    def get_estimate_x_0_vmap(self, observation_map):
        def estimate_x_0(x, t, timestep):
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            m = self.sqrt_alphas_cumprod[timestep]
            v = self.sqrt_1m_alphas_cumprod[timestep] ** 2
            s = self.score(x, t)
            x_0 = (x + v * s) / m
            return observation_map(x_0), (s, x_0)

        return estimate_x_0

    def get_estimate_x_0(self, observation_map):
        batch_observation_map = vmap(observation_map)

        def estimate_x_0(x, t, timestep):
            m = self.sqrt_alphas_cumprod[timestep]
            v = self.sqrt_1m_alphas_cumprod[timestep] ** 2
            s = self.score(x, t)
            x_0 = batch_mul(x + batch_mul(v, s), 1.0 / m)
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

    def get_estimate_x_0_vmap(self, observation_map):
        def estimate_x_0(x, t, timestep):
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            v = self.discrete_sigmas[timestep] ** 2
            s = self.score(x, t)
            x_0 = x + v * s
            return observation_map(x_0), (s, x_0)

        return estimate_x_0

    def get_estimate_x_0(self, observation_map):
        batch_observation_map = vmap(observation_map)

        def estimate_x_0(x, t, timestep):
            v = self.discrete_sigmas[timestep] ** 2
            s = self.score(x, t)
            x_0 = x + batch_mul(v, s)
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

    def __init__(self, model, eta=0.0, beta=None, ts=None):
        """
        Args:
            model: DDIM parameterizes the `epsilon(x, t) = -1. * fwd_marginal_std(t) * score(x, t)` function.
            eta: the hyperparameter for DDIM, a value of `eta=0.0` is deterministic 'probability ODE' solver, `eta=1.` is DDPMVP.
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

    def get_estimate_x_0_vmap(self, observation_map):
        def estimate_x_0(x, t, timestep):
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            m = self.sqrt_alphas_cumprod[timestep]
            sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
            epsilon = self.model(x, t)
            x_0 = (x - sqrt_1m_alpha * epsilon) / m
            return observation_map(x_0), (epsilon, x_0)

        return estimate_x_0

    def get_estimate_x_0(self, observation_map):
        batch_observation_map = vmap(observation_map)

        def estimate_x_0(x, t, timestep):
            m = self.sqrt_alphas_cumprod[timestep]
            sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
            epsilon = self.model(x, t)
            x_0 = batch_mul(x - batch_mul(sqrt_1m_alpha, epsilon), 1.0 / m)
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
        eta: the hyperparameter for DDIM, a value of `eta=0.0` is deterministic 'probability ODE' solver, `eta=1.` is DDPMVE.
    """

    def __init__(self, model, eta=0.0, sigma=None, ts=None):
        super().__init__(ts)
        if sigma is None:
            sigma = get_sigma_function(sigma_min=0.01, sigma_max=378.0)
        sigmas = vmap(sigma)(self.ts.flatten())
        self.sigma_max = sigmas[-1]
        self.discrete_sigmas = sigmas
        self.discrete_sigmas_prev = jnp.append(0.0, self.discrete_sigmas[:-1])
        self.eta = eta
        self.model = model

    def get_estimate_x_0_vmap(self, observation_map):
        def estimate_x_0(x, t, timestep):
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            std = self.discrete_sigmas[timestep]
            epsilon = self.model(x, t)
            x_0 = x - std * epsilon
            return observation_map(x_0), (epsilon, x_0)

        return estimate_x_0

    def get_estimate_x_0(self, observation_map):
        batch_observation_map = vmap(observation_map)

        def estimate_x_0(x, t, timestep):
            std = self.discrete_sigmas[timestep]
            epsilon = self.model(x, t)
            x_0 = x - batch_mul(std, epsilon)
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


# FROM TMPD


def batch_dot(a, b):
  return vmap(lambda a, b: a.T @ b)(a, b)


class STSL(DDIMVP):
  def __init__(self, scale, likelihood_strength, y, observation_map, adjoint_observation_map, noise_std, shape, model, eta=1.0, beta=None, ts=None):
    super().__init__(model, eta, beta, ts)
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map)
    self.y = y
    self.noise_std = noise_std
    # NOTE: Special case when num_y==1 can be handled correctly by defining observation_map output shape (1,)
    self.num_y = y.shape[1]
    self.num_x = sum([d for d in shape])
    self.observation_map = observation_map
    self.likelihood_score_vmap = self.get_likelihood_score_vmap()
    self.adjoint_observation_map = adjoint_observation_map
    self.batch_adjoint_observation_map = vmap(adjoint_observation_map)
    self.likelihood_strength = likelihood_strength
    self.scale = scale

  def get_likelihood_score_vmap(self):
    def l2_norm(x, t, timestep, y, rng):
      m = self.sqrt_alphas_cumprod[timestep]
      sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
      x = jnp.expand_dims(x, axis=0)
      t = jnp.expand_dims(t, axis=0)
      epsilon = self.model(x, t).squeeze(axis=0)
      x_0 = (x.squeeze(axis=0) - sqrt_1m_alpha * epsilon) / m
      score = - epsilon / sqrt_1m_alpha
      z = random.normal(rng, x.shape)
      score_perturbed = - self.model(x + z, t).squeeze(axis=0) / sqrt_1m_alpha
      h_x_0 = self.observation_map(x_0)
      norm = jnp.linalg.norm(y - h_x_0)
      scalar = - self.likelihood_strength * norm - (self.scale / self.num_x) * (jnp.dot(z, score_perturbed) - jnp.dot(z, score))
      return scalar[0], (score, x_0)

    grad_l2_norm = grad(l2_norm, has_aux=True)
    return vmap(grad_l2_norm, in_axes=(0, 0, 0, 0, None))

  def update(self, rng, x, t):
    timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
    beta = self.discrete_betas[timestep]
    sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
    v = sqrt_1m_alpha**2
    alpha = self.alphas[timestep]
    m_prev = self.sqrt_alphas_cumprod_prev[timestep]
    v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep]**2
    ls, (s, x_0) = self.likelihood_score_vmap(x, t, timestep, self.y, rng)
    x = x + ls
    x_mean = batch_mul(jnp.sqrt(alpha) * v_prev / v, x) + batch_mul(m_prev * beta / v, x_0)
    std = jnp.sqrt(beta * v_prev / v)
    rng, step_rng = random.split(rng)
    z = random.normal(step_rng, x.shape)
    x_mean = x_mean + batch_mul(std, z)
    return x, x_mean


class MPGD(DDIMVP):
  """
  TODO: This method requires a pretrained autoencoder.
  MPGD He, Murata et al. https://arxiv.org/pdf/2311.16424.pdf"""
  
  def __init__(self, scale, y, observation_map, noise_std, shape, model, eta=1.0, beta=None, ts=None):
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
    v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep]**2
    alpha_prev = m_prev**2
    x_0 = batch_mul((x - batch_mul(sqrt_1m_alpha, epsilon)), 1. / m)
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
    h_x_0, (epsilon, _) = self.estimate_h_x_0_vmap(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
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
    v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep]**2
    alpha_prev = m_prev**2
    coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha / alpha_prev))
    coeff2 = jnp.sqrt(v_prev - coeff1**2)
    posterior_score = - batch_mul(1. / sqrt_1m_alpha, epsilon) + ls
    x_mean = batch_mul(m_prev / m, x) + batch_mul(sqrt_1m_alpha * (sqrt_1m_alpha * m_prev / m - coeff2), posterior_score)
    std = coeff1
    return x_mean, std


class KGDMVPplus(KGDMVP):
  """KGDMVP with a mask."""
  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, vjp_h_x_0, (epsilon, _) = vjp(
        lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    C_yy = ratio * self.observation_map(vjp_h_x_0(self.observation_map(jnp.ones_like(x)))[0]) + self.noise_std**2
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return epsilon.squeeze(axis=0), ls


class KGDMVE(DDIMVE):
  def __init__(self, y, observation_map, noise_std, shape, model, eta=1.0, sigma=None, ts=None):
    super().__init__(model, eta, sigma, ts)
    self.eta = eta
    self.model = model
    self.discrete_sigmas = jnp.exp(
        jnp.linspace(jnp.log(self.sigma_min),
                      jnp.log(self.sigma_max),
                      self.num_steps))
    self.y = y
    self.noise_std = noise_std
    self.num_y = y.shape[1]
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map)
    self.batch_analysis_vmap = vmap(self.analysis)
    self.observation_map = observation_map
    self.batch_observation_map  = vmap(observation_map)
    self.axes_vmap = tuple(range(len(shape) + 1)[1:]) + (0,)

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, (epsilon, _) = self.estimate_h_x_0_vmap(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
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
    coeff2 = jnp.sqrt(sigma_prev**2  - coeff1**2)
    std = coeff1
    posterior_score = - batch_mul(1. / sigma, epsilon) + ls
    x_mean = x + batch_mul(sigma * (sigma - coeff2), posterior_score)
    return x_mean, std


class KGDMVEplus(KGDMVE):
  """KGDMVE with a mask."""
  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, vjp_h_x_0, (epsilon, _) = vjp(
        lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    C_yy = ratio * self.observation_map(vjp_h_x_0(self.observation_map(jnp.ones_like(x)))[0]) + self.noise_std**2
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return epsilon.squeeze(axis=0), ls


class PiGDMVP(DDIMVP):
  """PiGDM Song et al. 2023. Markov chain using the DDIM Markov Chain or VP SDE."""
  def __init__(self, y, observation_map, noise_std, shape, model, data_variance=1., eta=1., beta=None, ts=None):
    super().__init__(model, eta, beta, ts)
    # This method requires clipping in order to remain (robustly, over all samples) numerically stable
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map, clip=True, centered=True)
    self.batch_analysis_vmap = vmap(self.analysis)
    self.y = y
    self.noise_std = noise_std
    self.data_variance = data_variance
    self.num_y = y.shape[1]
    self.observation_map = observation_map

  def analysis(self, y, x, t, timestep, v, alpha):
    h_x_0, vjp_h_x_0, (epsilon, _) = vjp(
        lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    # Value suggested for VPSDE in original PiGDM paper
    r = v * self.data_variance  / (v + alpha * self.data_variance)
    C_yy = 1. + self.noise_std**2 / r
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
    v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep]**2
    alpha_prev = self.alphas_cumprod_prev[timestep]
    coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha / alpha_prev))
    coeff2 = jnp.sqrt(v_prev - coeff1**2)
    # TODO: slightly different to Algorithm 1
    posterior_score = - batch_mul(1. / sqrt_1m_alpha, epsilon) + ls
    x_mean = batch_mul(m_prev / m, x) + batch_mul(sqrt_1m_alpha * (sqrt_1m_alpha * m_prev / m - coeff2), posterior_score)
    std = coeff1
    return x_mean, std


class PiGDMVPplus(PiGDMVP):
  """PiGDMVP with a mask."""
  def analysis(self, y, x, t, timestep, v, alpha):
    h_x_0, vjp_estimate_h_x_0, (epsilon, _) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    # Value suggested for VPSDE in original PiGDM paper
    r = v * self.data_variance  / (v + alpha * self.data_variance)
    C_yy = 1. + self.noise_std**2 / r
    ls = vjp_estimate_h_x_0((y - h_x_0) / C_yy)[0]
    return epsilon.squeeze(axis=0), ls


class ReproducePiGDMVP(DDIMVP):
  """
  NOTE: We found this method to be unstable on CIFAR10 dataset, even with
    thresholding (clip=True) is used at each step of estimating x_0, and for each weighting
    schedule that we tried.
  PiGDM Song et al. 2023. Markov chain using the DDIM Markov Chain or VP SDE."""

  def __init__(self, y, observation_map, noise_std, shape, model, data_variance=1., eta=1., beta=None, ts=None):
    super().__init__(model, eta, beta, ts)
    self.data_variance = data_variance
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map, clip=True, centered=True)
    self.batch_analysis_vmap = vmap(self.analysis)
    self.y = y
    self.noise_std = noise_std
    self.num_y = y.shape[1]

  def analysis(self, y, x, t, timestep, v, alpha):
    h_x_0, vjp_estimate_h_x_0, (epsilon, x_0) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    # Value suggested for VPSDE in original PiGDM paper:
    r = v * self.data_variance  / (v + self.data_variance)
    # What it should really be set to, following the authors' mathematical reasoning:
    # r = v * self.data_variance  / (v + alpha * self.data_variance)
    C_yy = 1. + self.noise_std**2 / r
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
    v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep]**2
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
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    # Value suggested for VPSDE in original PiGDM paper:
    r = v * self.data_variance  / (v + self.data_variance)
    # What it should really be set to, following the authors' mathematical reasoning:
    # r = v * self.data_variance  / (v + alpha * self.data_variance)
    C_yy = 1. + self.noise_std**2 / r
    ls = vjp_estimate_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0), ls, epsilon.squeeze(axis=0)


class PiGDMVE(DDIMVE):
  """PiGDMVE for the SMLD Markov Chain or VE SDE."""
  def __init__(self, y, observation_map, noise_std, shape, model, data_variance=1., eta=1., sigma=None, ts=None):
    super().__init__(model, eta, sigma, ts)
    self.y = y
    self.data_variance = data_variance
    self.noise_std = noise_std
    self.num_y = y.shape[1]
    # This method requires clipping in order to remain (robustly, over all samples) numerically stable
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map, clip=True, centered=False)
    self.batch_analysis_vmap = vmap(self.analysis)

  def analysis(self, y, x, t, timestep, v):
    h_x_0, vjp_h_x_0, (epsilon, x_0) = vjp(
        lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    r = v * self.data_variance / (v + self.data_variance)
    C_yy = 1. + self.noise_std**2 / r
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0), ls, epsilon.squeeze(axis=0)

  def posterior(self, x, t):
    timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
    sigma = self.discrete_sigmas[timestep]
    sigma_prev = self.discrete_sigmas_prev[timestep]
    x_0, ls, epsilon = self.batch_analysis_vmap(self.y, x, t, timestep, sigma**2)
    coeff1 = self.eta * jnp.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
    coeff2 = jnp.sqrt(sigma_prev**2  - coeff1**2)
    x_mean = x_0 + batch_mul(coeff2, epsilon) + ls
    std = coeff1
    return x_mean, std


class PiGDMVEplus(PiGDMVE):
  """KGDMVE with a mask."""
  def analysis(self, y, x, t, timestep, v):
    h_x_0, vjp_h_x_0, (epsilon, x_0) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    # Value suggested for VPSDE in original PiGDM paper
    r = v * self.data_variance  / (v + self.data_variance)
    C_yy = 1. + self.noise_std**2 / r
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
      self.get_estimate_x_0(observation_map, clip=True, centered=False))
    self.likelihood_score_vmap = self.get_likelihood_score_vmap(
      self.get_estimate_x_0_vmap(observation_map, clip=True, centered=False))

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
      self.get_estimate_x_0(observation_map, clip=True, centered=True))
    self.likelihood_score_vmap = self.get_likelihood_score_vmap(
      self.get_estimate_x_0_vmap(observation_map, clip=True, centered=True))

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
    h_x_0, (_, x_0) = self.estimate_h_x_0_vmap(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
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
    v = self.sqrt_1m_alphas_cumprod[timestep]**2
    ratio = v / m
    x_0 = self.batch_analysis_vmap(self.y, x, t, timestep, ratio)
    alpha = self.alphas[timestep]
    m_prev = self.sqrt_alphas_cumprod_prev[timestep]
    v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep]**2
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
          lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
      C_yy = self.observation_map(vjp_estimate_h_x_0(self.observation_map(jnp.ones_like(x)))[0]) + self.noise_std**2 / ratio
      ls = vjp_estimate_h_x_0((y - h_x_0) / C_yy)[0]
      return x_0.squeeze(axis=0) + ls


class KPSMLD(SMLD):
  """Kalman posterior for SMLD Ancestral sampling."""
  def __init__(self, y, observation_map, noise_std, shape, score, sigma=None, ts=None):
    super().__init__(score, sigma, ts)
    self.y = y
    self.noise_std = noise_std
    self.num_y = y.shape[1]
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map, clip=True, centered=False)
    self.batch_analysis_vmap = vmap(self.analysis)
    self.observation_map = observation_map
    self.batch_observation_map = vmap(observation_map)
    self.axes_vmap = tuple(range(len(shape) + 1)[1:]) + (0,)

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, (_, x_0) = self.estimate_h_x_0_vmap(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
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
    x_mean = batch_mul(sigma_prev**2 / sigma**2, x) + batch_mul(1 - sigma_prev**2 / sigma**2, x_0)
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
        lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    C_yy = self.observation_map(vjp_h_x_0(self.observation_map(jnp.ones_like(x)))[0]) + self.noise_std**2 / ratio
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0) + ls


class KPSMLDdiag(KPSMLD):
  """Kalman posterior for SMLD Ancestral sampling."""

  # def get_grad_estimate_x_0_vmap(self, observation_map):
  #   """https://stackoverflow.com/questions/70956578/jacobian-diagonal-computation-in-jax"""

  #   def estimate_x_0_single(val, i, x, t, timestep):
  #     x_shape = x.shape
  #     x = x.flatten()
  #     x = x.at[i].set(val)
  #     x = x.reshape(x_shape)
  #     x = jnp.expand_dims(x, axis=0)
  #     t = jnp.expand_dims(t, axis=0)
  #     v = self.discrete_sigmas[timestep]**2
  #     s = self.score(x, t)
  #     x_0 = x + v * s
  #     h_x_0 = observation_map(x_0)
  #     return h_x_0[i]
  #   return vmap(value_and_grad(estimate_x_0_single), in_axes=(0, 0, None, None, None))

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, vjp_h_x_0, (_, x_0) = vjp(
        lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)

    # There is no natural way to do this with JAX's transforms:
    # you cannot map the input, because in general each diagonal entry of the jacobian depends on all inputs.
    # This seems like the best method, but it is too slow for numerical evaluation, and batch size cannot be large (max size tested was one)
    vec_vjp_h_x_0 = vmap(vjp_h_x_0)
    diag = jnp.diag(self.batch_observation_map(vec_vjp_h_x_0(jnp.eye(y.shape[0]))[0]))

    # # This method gives OOM error
    # idx = jnp.arange(len(y))
    # h_x_0, diag = self.grad_estimate_x_0_vmap(x.flatten(), idx, x, t, timestep)
    # diag = self.observation_map(diag)

    # # This method can't be XLA compiled and is way too slow for numerical evaluation
    # diag = jnp.empty(y.shape[0])
    # for i in range(y.shape[0]):
    #   eye = jnp.zeros(y.shape[0])
    #   diag_i = jnp.dot(self.observation_map(vjp_h_x_0(eye)[0]), eye)
    #   diag = diag.at[i].set(diag_i)

    C_yy = diag + self.noise_std**2 / ratio
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0) + ls


class KPDDPMdiag(KPDDPM):
  """Kalman posterior for DDPM Ancestral sampling."""

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, vjp_h_x_0, (_, x_0) = vjp(
        lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    vec_vjp_h_x_0 = vmap(vjp_h_x_0)
    diag = jnp.diag(self.batch_observation_map(vec_vjp_h_x_0(jnp.eye(y.shape[0]))[0]))
    C_yy = diag + self.noise_std**2 / ratio
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0) + ls
