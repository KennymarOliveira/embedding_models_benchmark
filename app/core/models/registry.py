import logging
from typing import Any, Dict, Type

from app.core.config import AVAILABLE_MODELS, get_model_config
from app.core.models.base import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registro central de modelos de embedding suportados."""

    _registry: Dict[str, Type[BaseEmbeddingModel]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorador para registrar uma nova classe de modelo."""

        def decorator(subclass: Type[BaseEmbeddingModel]):
            cls._registry[name.lower()] = subclass
            return subclass

        return decorator

    @classmethod
    def get_model_class(cls, class_key: str) -> Type[BaseEmbeddingModel]:
        """Obtém a classe registrada pelo class_key."""
        key = class_key.lower()
        if key not in cls._registry:
            raise KeyError(
                f"Classe de modelo '{class_key}' não está registrada. "
                f"Classes disponíveis: {list(cls._registry.keys())}"
            )
        return cls._registry[key]

    @classmethod
    def create_model(
        cls,
        model_name: str,
        custom_config: Dict[str, Any] = None,
    ) -> BaseEmbeddingModel:
        """
        Instancia um modelo a partir do seu identificador em config.py
        ou com configurações customizadas.
        """
        base_cfg = get_model_config(model_name).copy()
        if custom_config:
            base_cfg.update(custom_config)

        class_key = base_cfg.get("class_key", "generic_st")
        model_cls = cls.get_model_class(class_key)
        return model_cls(base_cfg)

    @classmethod
    def list_registered_classes(cls) -> list[str]:
        """Lista todas as chaves de classes de modelos registradas."""
        return sorted(list(cls._registry.keys()))

