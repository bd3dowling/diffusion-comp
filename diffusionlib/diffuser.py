from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as rand
from jax import lax
from jaxtyping import Array, PRNGKeyArray

from diffusionlib.mean_processor import MeanProcessor, MeanProcessorType
from diffusionlib.utils import extract_and_expand
from diffusionlib.variance_processor import VarianceProcessor, VarianceProcessorType


@dataclass
class DDPM:
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
    ) -> DDPM:
        betas = jnp.array(
            [beta_min / n + i / (n * (n - 1)) * (beta_max - beta_min) for i in range(n)]
        )

        return cls(betas=betas, **kwargs)

    def q_sample(self, x_0: Array, t: Array, key: PRNGKeyArray):
        """Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).
        """
        noise = rand.normal(key, x_0.shape)
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
            key, subkey1, subkey2 = rand.split(key, 3)

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

        noise = rand.normal(key, x.shape)
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
