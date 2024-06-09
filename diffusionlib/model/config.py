"""Pydantic models for configs."""

from importlib.abc import Traversable
from importlib.resources import files
from typing import Any

import yaml
from pydantic import BaseModel
from strenum import StrEnum

import config
import config.model
import data.model_checkpoint
import data.samples


class ModelName(StrEnum):
    FFHQ = "ffhq"
    IMAGENET = "imagenet"


class ModelConfig(BaseModel, frozen=True):
    image_size: int
    num_channels: int
    num_res_blocks: int
    channel_mult: str
    learn_sigma: bool
    class_cond: bool
    use_checkpoint: bool
    attention_resolutions: str | int
    num_heads: int
    num_head_channels: int
    num_heads_upsample: int
    use_scale_shift_norm: bool
    dropout: int
    resblock_updown: bool
    use_new_attention_order: bool
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
