"""Pydantic models for configs."""

from enum import StrEnum, auto
from importlib.abc import Traversable
from importlib.resources import files
from typing import Any

import yaml
from pydantic import BaseModel, Field
from typing_extensions import Annotated

import config
import config.model
import data.model_checkpoint


class ModelName(StrEnum):
    FFHQ = auto()
    IMAGENET = auto()
    NCSNPP = auto()
    MLP = auto()


class ModelConfig(BaseModel):
    attention_resolutions: str | int
    channel_mult: str
    class_cond: bool
    dropout: int
    image_size: int
    learn_sigma: bool
    num_channels: int
    num_head_channels: int
    num_heads: int
    num_heads_upsample: int
    num_res_blocks: int
    resblock_updown: bool
    use_checkpoint: bool
    use_new_attention_order: bool
    use_scale_shift_norm: bool
    _model_name: ModelName  # Not validated

    @classmethod
    def load(cls, model_name: ModelName) -> "ModelConfig":
        model_conf_path = files(config.model) / f"{model_name}.yaml"

        with model_conf_path.open() as f:
            model_conf_raw: dict[str, Any] = yaml.safe_load(f)

        instance = cls(**model_conf_raw)
        instance._model_name = model_name

        return instance

    @property
    def model_checkpoint_path(self) -> Traversable:
        return files(data.model_checkpoint) / f"{self._model_name}.pt"


class _AttentionType(StrEnum):
    DDPM = auto()


class _EmbeddingType(StrEnum):
    FOURIER = auto()
    POSITIONAL = auto()


class _NonLinearity(StrEnum):
    SWISH = auto()


class _Normalization(StrEnum):
    INSTANCE_NORM = auto()
    INSTANCE_NORM_PLUS = auto()
    INSTANCE_NORM_PLUS_PLUS = auto()
    VARIANCE_NORM = auto()
    GROUP_NORM = auto()


class _Progressive(StrEnum):
    OUTPUT_SKIP = auto()
    NONE = auto()
    RESIDUAL = auto()


class _ProgressiveInput(StrEnum):
    INPUT_SKIP = auto()
    NONE = auto()
    RESIDUAL = auto()


class _ProgressiveCombine(StrEnum):
    CAT = auto()
    SUM = auto()


class _ResblockType(StrEnum):
    BIGGAN = auto()
    DDPM = auto()


class JaxModelConfig(BaseModel):
    attention_type: _AttentionType
    attn_resolutions: list[int]
    beta_max: Annotated[float, Field(strict=True, gt=0)] = 1
    beta_min: Annotated[float, Field(strict=True, gt=0)] = 1
    ch_mult: list[int]
    conditional: bool
    conv_size: int
    dropout: Annotated[float, Field(strict=True, gt=0, lt=1)]
    ema_rate: Annotated[float, Field(strict=True, gt=0, lt=1)]
    embedding_type: _EmbeddingType
    fir: bool
    fir_kernel: list[int]
    fourier_scale: int = 0
    init_scale: float
    name: ModelName
    nf: int
    nonlinearity: _NonLinearity
    normalization: _Normalization
    num_res_blocks: int
    num_scales: int
    progressive: _Progressive
    progressive_combine: _ProgressiveCombine
    progressive_input: _ProgressiveInput
    resamp_with_conv: bool
    resblock_type: _ResblockType
    scale_by_sigma: bool
    sigma_max: Annotated[float, Field(strict=True, gt=0)] = 1
    sigma_min: Annotated[float, Field(strict=True, gt=0)] = 1
    skip_rescale: bool
