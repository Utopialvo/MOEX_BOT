# Файл: app/api/v1/system.py
"""Системные эндпоинты: здоровье, метрики, информация."""
import os
from fastapi import APIRouter
from app.services.db import db_client
from app.services.model import model_service
from app.workers.task_manager import task_manager
from app.models.schemas import HealthResponse, MetricsResponse, InfoResponse
from app.config import settings
from app.core.constants import SUPPORTED_INTERVALS

router = APIRouter(prefix="/system", tags=["System"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Проверяет подключение к ClickHouse и наличие хотя бы одной модели."""
    ch_ok = True
    try:
        db_client._get_client().ping()
    except Exception:
        ch_ok = False
    try:
        _ = model_service.default_model
        model_ok = True
    except Exception:
        model_ok = False
    status = "ok" if (ch_ok and model_ok) else "degraded"
    return HealthResponse(status=status, clickhouse=ch_ok, model_loaded=model_ok)


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    """Возвращает текущие метрики системы."""
    tasks = task_manager.list_tasks()
    active = sum(1 for t in tasks if t["status"] == "running")
    total_candles = None
    try:
        res = db_client._get_client().query_df(
            "SELECT sum(rows) FROM system.parts WHERE database='default' AND table LIKE 'raw_candles_%'"
        )
        if not res.empty:
            total_candles = int(res.iloc[0, 0])
    except Exception:
        pass
    last_error = next((t["error_message"] for t in tasks if t.get("error_message")), None)
    return MetricsResponse(
        active_tasks=active,
        total_tasks=len(tasks),
        total_candles=total_candles,
        last_error=last_error,
    )


@router.get("/info", response_model=InfoResponse)
def get_info() -> InfoResponse:
    """Метаданные для агентов: активные тикеры, модели, настройки."""
    active_tickers = list(set(
        t["ticker"] for t in task_manager.list_tasks() if t["status"] == "running"
    ))
    models = []
    if os.path.exists(settings.DEFAULT_MODEL_PATH):
        models.append({"name": "default", "path": settings.DEFAULT_MODEL_PATH})
    model_dir = settings.MODEL_DIR
    if os.path.isdir(model_dir):
        for f in os.listdir(model_dir):
            if f.endswith(".cbm"):
                name = f[:-4]
                models.append({"name": name, "path": os.path.join(model_dir, f)})
    pauses = {
        "1m": settings.PAUSE_1M,
        "10m": settings.PAUSE_10M,
    }
    return InfoResponse(
        active_tickers=active_tickers,
        supported_intervals=list(SUPPORTED_INTERVALS.keys()),
        available_models=models,
        pauses=pauses,
    )