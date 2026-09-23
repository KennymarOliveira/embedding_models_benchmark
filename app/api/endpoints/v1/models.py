from typing import List

from fastapi import APIRouter, HTTPException

from app.core.config import AVAILABLE_MODELS, get_model_config
from app.schemas.benchmark import ModelConfigResponse

router = APIRouter()


@router.get("/", response_model=List[ModelConfigResponse])
def list_models():
    """Lista todos os modelos de embedding locais configurados no sistema."""
    models_list = []
    for key, cfg in AVAILABLE_MODELS.items():
        models_list.append(
            ModelConfigResponse(
                key=key,
                model_id=cfg["model_id"],
                display_name=cfg["display_name"],
                class_key=cfg.get("class_key", "generic_st"),
                enabled=cfg.get("enabled", True),
                device=str(cfg.get("device", "auto")),
                torch_dtype=str(cfg.get("torch_dtype", "float32")),
                max_seq_length=cfg.get("max_seq_length", 512),
                default_batch_size=cfg.get("default_batch_size", 8),
                description=cfg.get("description", ""),
            )
        )
    return models_list


@router.get("/{model_name}", response_model=ModelConfigResponse)
def get_model_details(model_name: str):
    """Retorna detalhes e parâmetros de configuração de um modelo específico."""
    try:
        cfg = get_model_config(model_name)
        target_key = model_name
        for k, v in AVAILABLE_MODELS.items():
            if v == cfg:
                target_key = k
                break

        return ModelConfigResponse(
            key=target_key,
            model_id=cfg["model_id"],
            display_name=cfg["display_name"],
            class_key=cfg.get("class_key", "generic_st"),
            enabled=cfg.get("enabled", True),
            device=str(cfg.get("device", "auto")),
            torch_dtype=str(cfg.get("torch_dtype", "float32")),
            max_seq_length=cfg.get("max_seq_length", 512),
            default_batch_size=cfg.get("default_batch_size", 8),
            description=cfg.get("description", ""),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

