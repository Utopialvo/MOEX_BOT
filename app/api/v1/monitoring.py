# Файл: app/api/v1/monitoring.py
"""
Эндпоинты для отслеживания качества прогнозов.
Метрики считаются на точном совпадении timestamp предсказания и begin свечи.
Все обработчики синхронные.
"""
import pandas as pd
from fastapi import APIRouter, Query, HTTPException
from app.models.schemas import ModelQualityResponse, ModelQualityMetrics, ModelQualityHistoryResponse
from app.services.db import db_client
from app.core.constants import SUPPORTED_INTERVALS, get_predictions_table, get_raw_candles_table

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


def _check_tables(ticker: str, interval: str) -> None:
    pred_table = get_predictions_table(ticker, interval)
    raw_table = get_raw_candles_table(ticker, interval)
    if not db_client.table_exists(pred_table) or not db_client.table_exists(raw_table):
        raise HTTPException(
            404,
            "Required tables not found. Start a prediction task first."
        )


@router.get("/model_quality", response_model=ModelQualityResponse)
def get_model_quality(
    ticker: str,
    interval: str = Query(..., description="Интервал (1m, 10m)"),
    window: int = Query(20, ge=1, le=500, description="Количество последних окон для усреднения"),
) -> ModelQualityResponse:
    """Оценка качества модели на последних данных."""
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(400, "Unsupported interval")
    _check_tables(ticker, interval)

    pred_table = get_predictions_table(ticker, interval)
    raw_table = get_raw_candles_table(ticker, interval)

    query = f"""
        SELECT
            p.timestamp,
            p.value,
            c.close
        FROM {pred_table} p
        JOIN {raw_table} c ON p.timestamp = c.begin
        WHERE p.ticker = '{ticker}' AND p.interval = '{interval}'
        ORDER BY p.timestamp DESC
        LIMIT {window}
    """
    try:
        df = db_client._get_client().query_df(query)
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")

    if df.empty:
        raise HTTPException(404, "No matching predictions and candles found")

    errors = df["value"] - df["close"]
    mae = errors.abs().mean()
    rmse = (errors ** 2).mean() ** 0.5
    mape = (errors.abs() / df["close"]).mean() * 100
    last_eval = df["timestamp"].max()

    status = "ok" if rmse < 5.0 else "degraded"
    return ModelQualityResponse(
        ticker=ticker,
        interval=interval,
        last_evaluation_time=last_eval,
        window_size=len(df),
        metrics=ModelQualityMetrics(
            rmse=round(rmse, 4),
            mape=round(mape, 2),
            mae=round(mae, 4),
        ),
        status=status,
    )


@router.get("/model_quality/history", response_model=ModelQualityHistoryResponse)
def get_model_quality_history(
    ticker: str,
    interval: str = Query(..., description="Интервал (1m, 10m)"),
    limit: int = Query(50, ge=1, le=1000),
) -> ModelQualityHistoryResponse:
    """История качества модели, сгруппированная по дням."""
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(400, "Unsupported interval")
    _check_tables(ticker, interval)

    pred_table = get_predictions_table(ticker, interval)
    raw_table = get_raw_candles_table(ticker, interval)

    query = f"""
        SELECT
            p.timestamp,
            p.value,
            c.close
        FROM {pred_table} p
        JOIN {raw_table} c ON p.timestamp = c.begin
        WHERE p.ticker = '{ticker}' AND p.interval = '{interval}'
        ORDER BY p.timestamp DESC
        LIMIT {limit}
    """
    try:
        df = db_client._get_client().query_df(query)
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")

    history = []
    if not df.empty:
        df["date"] = df["timestamp"].dt.date
        for date, group in df.groupby("date"):
            errors = group["value"] - group["close"]
            mae = errors.abs().mean()
            rmse = (errors ** 2).mean() ** 0.5
            mape = (errors.abs() / group["close"]).mean() * 100
            history.append(
                ModelQualityResponse(
                    ticker=ticker,
                    interval=interval,
                    last_evaluation_time=group["timestamp"].max(),
                    window_size=len(group),
                    metrics=ModelQualityMetrics(
                        rmse=round(rmse, 4),
                        mape=round(mape, 2),
                        mae=round(mae, 4),
                    ),
                    status="ok",
                )
            )

    return ModelQualityHistoryResponse(history=history[:limit])