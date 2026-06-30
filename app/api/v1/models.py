# Файл: MOEX_BOT/app/api/v1/models.py
"""Управление загруженными моделями."""
import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.config import settings
from app.services.model import model_service
from app.core.logging import logger

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("")
def list_models() -> dict:
    """Возвращает список доступных моделей (загруженных или на диске)."""
    models = []
    if os.path.exists(settings.DEFAULT_MODEL_PATH):
        models.append({"ticker": "default", "path": settings.DEFAULT_MODEL_PATH})
    model_dir = settings.MODEL_DIR
    if os.path.isdir(model_dir):
        for f in os.listdir(model_dir):
            if f.endswith(".cbm"):
                ticker = f[:-4]   # убираем расширение
                models.append({"ticker": ticker, "path": os.path.join(model_dir, f)})
    return {"models": models}


@router.post("/upload/{ticker}")
async def upload_model(ticker: str, file: UploadFile = File(...)) -> dict:
    """Загружает новый файл модели для тикера."""
    if not file.filename.endswith(".cbm"):
        raise HTTPException(400, "Only .cbm files allowed")
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    # Сохраняем как {ticker}.cbm
    path = os.path.join(settings.MODEL_DIR, f"{ticker}.cbm")
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)
    # Инвалидируем кэш для этого тикера
    if ticker in model_service.models:
        del model_service.models[ticker]
    # Сброс дефолтной модели, если путь совпадает (на случай перезаписи default)
    if path == settings.DEFAULT_MODEL_PATH:
        model_service._default_model = None
    logger.info(f"Model for {ticker} uploaded to {path}")
    return {"message": f"Model for {ticker} saved"}


@router.delete("/{ticker}")
def delete_model(ticker: str) -> dict:
    """Удаляет модель тикера с диска (дефолтную удалить нельзя)."""
    if ticker == "default":
        raise HTTPException(400, "Cannot delete default model")
    path = os.path.join(settings.MODEL_DIR, f"{ticker}.cbm")
    if not os.path.exists(path):
        raise HTTPException(404, "Model not found")
    os.remove(path)
    if ticker in model_service.models:
        del model_service.models[ticker]
    return {"message": f"Model for {ticker} deleted"}