from dataclasses import dataclass, field

import numpy as np
from jax import random
from jaxtyping import Array, PRNGKeyArray
from particles.core import SMC, FeynmanKac

from diffusionlib.sampler.base import Sampler, SamplerName, register_sampler


@register_sampler(name=SamplerName.SMC)
@dataclass(kw_only=True)
class SMCSampler(Sampler):
    num_steps: int = 1000
    num_particles: int = 1000
    essr_min: float
    resampling: str = "systematic"  # TODO: full typing
    fk_model: FeynmanKac
    smc: SMC = field(init=False)
    particle_history: Array = field(init=False)

    def __post_init__(self) -> None:
        self.smc = SMC(
            fk=self.fk_model,
            N=self.num_particles,
            ESSrmin=self.essr_min,
            resampling=self.resampling,
            store_history=True,
        )

    def sample(self, rng: PRNGKeyArray, x_0: Array | None = None) -> Array:
        rng, sub_rng = random.split(rng)
        np.random.seed(sub_rng[0])

        self.smc.run()

        final_particles: Array = self.smc.X
        self.particle_history = self.smc.hist.X

        return final_particles
