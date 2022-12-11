import abc
import importlib.util
import logging
import sys
from functools import partial
from pathlib import Path
from typing import (
    Dict,
    List,
    Type,
)

import numpy as np
import yaml

__all__ = ["Estimator", "discover_models"]


DUMPS_FOLDER = Path(__file__).parent.parent.joinpath("dumps")


class EstimatorMeta(type(yaml.YAMLObject)):  # type: ignore
    """
    Estimator metaclass, automatically passes
    name, yaml_tag, and yaml_loader to estimators.
    """

    def __new__(cls, name, bases, attrs):
        if "yaml_tag" not in attrs:
            attrs["yaml_tag"] = f"!{name}"
        attrs["yaml_loader"] = yaml.SafeLoader
        return super().__new__(cls, name, bases, attrs)


class Estimator(yaml.YAMLObject, metaclass=EstimatorMeta):  # type: ignore
    """
    Recommendation estimator base class.
    """

    @classmethod
    def from_yaml(cls, loader, node):
        """Load estimator from yaml"""
        kwargs = loader.construct_mapping(node)
        name = kwargs.get("name", None) or cls.__name__
        return {name: partial(cls, **kwargs)}

    def __init__(self, name=None):
        super().__init__()
        self.name = name or self.__class__.__name__

    @abc.abstractmethod
    def recommend(self, user_ids: List[int], n: int) -> np.ndarray:
        """
        Recommend items for users.

        Args:
            user_ids: user ids
            n: number of recommendations

        Returns:
            np.ndarray: array of recommendations
        """

        raise NotImplementedError


def load_estimators(manifests) -> Dict[str, Type[Estimator]]:
    estimators = {}

    for manifest in manifests:
        with open(manifest, "r") as f:
            raw = yaml.safe_load(f)
        for estimator in raw:
            estimators.update(estimator)

    return estimators


log = logging.getLogger(__name__)


def discover_models() -> Dict[str, Type[Estimator]]:
    """Discover all models in model path"""

    models_directory = Path(__file__).parent.joinpath("models")

    for path in models_directory.glob("*.py"):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if not spec:
            log.warning(f"Cannot import {path}")
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module

        try:
            spec.loader.exec_module(module)  # type: ignore
        except Exception as e:
            log.warning(f"Cannot import {path}: {e}")
            continue

    estimators = load_estimators(
        [Path(__file__).parent.joinpath("manifest.yaml")]
    )

    return estimators
