# Файл: MOEX_BOT/app/models/schemas.py
"""Pydantic-схемы для запросов и ответов."""

from datetime import datetime
from typing import Optional, List, Literal, Any
from uuid import UUID
from pydantic import BaseModel, Field


class TaskCreate(BaseModel):
    """Запрос на создание задачи."""
    ticker: str = Field(..., description="Тикер инструмента (GAZP, SBER)")
    interval: Literal["1m", "10m"] = Field(..., description="Интервал свечей")
    model_name: Optional[str] = Field(None, description="Имя файла модели (без расширения) в /app/models")


class BulkTaskCreate(BaseModel):
    """Массовое создание задач."""
    tasks: List[TaskCreate]


class TaskInfo(BaseModel):
    """Информация о задаче."""
    task_id: UUID
    ticker: str
    interval: str
    status: Literal["running", "stopped", "error", "pending"]
    created_at: datetime
    last_run: Optional[datetime] = None
    error_message: Optional[str] = None
    model_name: Optional[str] = None
    recalc_status: Literal["idle", "in_progress", "completed", "failed"] = "idle"


class TaskListResponse(BaseModel):
    """Список задач."""
    tasks: List[TaskInfo]
    total: int


class CandleResponse(BaseModel):
    """Одна свеча."""
    begin: datetime
    open: float
    high: float
    low: float
    close: float
    value: float
    volume: int


class CandleListResponse(BaseModel):
    """Ответ со списком свечей."""
    candles: List[CandleResponse]
    ticker: str
    interval: str


class PredictionResponse(BaseModel):
    """Одно предсказание."""
    timestamp: datetime
    value: float
    ticker: str
    interval: str


class PredictionHistoryResponse(BaseModel):
    """История предсказаний."""
    predictions: List[PredictionResponse]


class AggregateResponse(BaseModel):
    """Агрегированные данные за период."""
    timestamp: datetime
    avg_price: float
    max_price: float
    min_price: float
    total_volume: int
    count: int


class SignalResponse(BaseModel):
    """Торговый сигнал."""
    begin: datetime
    buy_signal: int
    sell_signal: int


class SQLQueryRequest(BaseModel):
    """Тело запроса для выполнения SQL."""
    query: str = Field(..., description="SQL-запрос (только SELECT)")


class SQLQueryResponse(BaseModel):
    """Результат выполнения SQL."""
    columns: List[str]
    rows: List[List[Any]]
    row_count: int


class HealthResponse(BaseModel):
    """Ответ проверки здоровья."""
    status: str
    clickhouse: bool
    model_loaded: bool
    details: Optional[dict] = None


class MetricsResponse(BaseModel):
    """Системные метрики."""
    active_tasks: int
    total_tasks: int
    total_candles: Optional[int] = None
    last_error: Optional[str] = None


class ModelQualityMetrics(BaseModel):
    """Значения метрик качества."""
    rmse: float
    mape: float
    mae: Optional[float] = None


class ModelQualityResponse(BaseModel):
    """Качество модели на одном окне."""
    ticker: str
    interval: str
    last_evaluation_time: Optional[datetime]
    window_size: int
    metrics: ModelQualityMetrics
    status: str = "ok"


class ModelQualityHistoryResponse(BaseModel):
    """История качества модели."""
    history: List[ModelQualityResponse]


# ---- Новые модели для batch-запросов и метаданных ----

class BatchPredictionRequest(BaseModel):
    tickers: List[str]
    interval: Literal["1m", "10m"]


class BatchPredictionResponse(BaseModel):
    predictions: dict[str, PredictionResponse]  # тикер -> предсказание


class BatchCandleRequest(BaseModel):
    tickers: List[str]
    interval: Literal["1m", "10m"]
    limit: int = Field(1000, ge=1, le=10000)


class BatchCandleResponse(BaseModel):
    candles: dict[str, List[CandleResponse]]  # тикер -> список свечей


class InfoResponse(BaseModel):
    api_version: str = "1.2.0"
    active_tickers: List[str]
    supported_intervals: List[str]
    available_models: List[dict]
    pauses: dict