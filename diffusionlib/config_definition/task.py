# TODO: Not this
from importlib.resources import files
from typing import Any

import yaml
from pydantic import BaseModel

from config import task
from diffusionlib.conditioning_method import ConditioningMethodName


class _TrainingConfig(BaseModel):
    sde: str
    n_iters: int
    batch_size: int
    likelihood_weighting: bool
    score_scaling: bool
    reduce_mean: bool
    log_epoch_freq: int | None = None
    log_step_freq: int
    pmap: bool
    n_jitted_steps: int
    snapshot_freq: int
    snapshot_freq_for_preemption: int
    eval_freq: int
    continuous: bool
    pointwise_t: bool


class _EvalConfig(BaseModel):
    batch_size: int
    ckpt_id: int
    enable_sampling: int
    num_samples: int
    enable_loss: bool
    enable_bpd: bool
    bpd_dataset: str
    pmap: bool = True


class _SamplingConfig(BaseModel):
    cs_method: ConditioningMethodName
    noise_std: float
    denoise: bool
    innovation: bool
    inverse_scaler: str | None
    stack_samples: bool
    method: str
    predictor: str
    corrector: str
    n_steps_each: int
    noise_removal: bool
    probability_flow: bool
    snr: float
    projection_sigma_rate: float
    cs_solver: str
    expansion: int
    coeff: float
    n_projections: int
    task: str
    lambd: float
    denoise_override: bool


class _DataConfig(BaseModel):
    image_size: int
    num_channels: int | None
    random_flip: bool
    uniform_dequantization: bool
    centered: bool


class _ModelConfig(BaseModel):
    beta_min: float
    beta_max: float
    sigma_min: float
    sigma_max: float
    name: str
    num_scales: int
    dropout: float
    embedding_type: str


class _SolverConfig(BaseModel):
    num_outer_steps: int
    outer_solver: str
    eta: float | None = None
    inner_solver: str | None
    stsl_scale_hyperparameter: float | None = None
    dps_scale_hyperparameter: float | None = None
    num_inner_steps: int
    dt: None
    epsilon: None
    snr: None


class _OptimConfig(BaseModel):
    optimizer: str
    lr: float
    warmup: bool | int
    weight_decay: bool
    grad_clip: float | None
    seed: int = 0
    beta1: float
    eps: float


class TaskConfig(BaseModel):
    training: _TrainingConfig
    eval: _EvalConfig
    sampling: _SamplingConfig
    data: _DataConfig
    model: _ModelConfig
    solver: _SolverConfig
    optim: _OptimConfig
    seed: int

    @classmethod
    def load(cls, name: str = "gmm") -> "TaskConfig":
        task_conf_path = files(task) / f"{name}.yaml"

        with task_conf_path.open() as f:
            task_conf_raw: dict[str, Any] = yaml.safe_load(f)

        return cls(**task_conf_raw)
