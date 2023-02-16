"""The configuration module. It loads and creates models, optimizers, 
and schedulers from a configuration file.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Type

import torch.nn
import torch.optim

from img2img.models import reprojector
from img2img.utils import NullScheduler


@dataclass
class LoadedModel:
    """A loaded model."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    loss_fn: torch.nn.Module


OPTIMIZERS_MAP = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}

SCHEDULERS_MAP = {
    "step": torch.optim.lr_scheduler.StepLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "null": NullScheduler,
}

LOSS_FN_MAP = {
    "mse": torch.nn.MSELoss,
    "cos": torch.nn.CosineSimilarity,
}


def build_from_config(object_type: Type, cfg: dict) -> Any:
    """Build an object from a configuration dictionary.

    Parameters
    ----------
    object_type : Type
        The type of the object to build.
    cfg : dict
        The configuration dictionary.

    Returns
    -------
    Any
        The built object.
    """

    args = cfg["args"]
    kwargs = cfg["kwargs"]

    return object_type(*args, **kwargs)


def load_model(config_path: str) -> tuple[LoadedModel, dict]:
    """Load a model from a configuration file.

    Parameters
    ----------
    config_path : str
        The path to the configuration file.

    Returns
    -------
    LoadedModel
        The loaded model.
    dict
        The configuration dictionary.
    """

    with open(config_path, "r") as handle:
        config = json.load(handle)

    model_related_config = config["model"]
    model_class = getattr(reprojector, model_related_config["class"])
    model = build_from_config(model_class, model_related_config)

    optimizer_related_config = config["optimizer"]
    optimizer_class = OPTIMIZERS_MAP[optimizer_related_config["class"]]
    optimizer = optimizer_class(
        model.parameters(), **optimizer_related_config["kwargs"]
    )

    scheduler_related_config = config["scheduler"]
    scheduler_class = SCHEDULERS_MAP[scheduler_related_config["class"]]
    scheduler = scheduler_class(optimizer, **scheduler_related_config["kwargs"])

    loss_fn_related_config = config["loss_fn"]
    loss_fn_class = LOSS_FN_MAP[loss_fn_related_config["class"]]
    loss_fn = loss_fn_class(**loss_fn_related_config["kwargs"])

    return LoadedModel(model, optimizer, scheduler, loss_fn), config
