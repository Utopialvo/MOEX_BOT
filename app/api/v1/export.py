# Файл: app/api/v1/export.py
"""Экспорт данных в CSV."""
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
import io
from app.services.db import db_client
from app.core.constants import SUPPORTED_INTERVALS, get_raw_candles_table

router = APIRouter(prefix="/export", tags=["Export"])


@router.get("/candles/{ticker}")
def export_candles(
    ticker: str,
    interval: str = Query(..., description="Интервал (1m, 10m)"),
    from_date: str | None = None,
    to_date: str | None = None,
    format: str = "csv",
) -> StreamingResponse:
    """Экспорт исторических свечей в CSV."""
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(400, "Unsupported interval")
    table = get_raw_candles_table(ticker, interval)
    if not db_client.table_exists(table):
        raise HTTPException(
            404,
            f"Table '{table}' not found. "
            f"Start a prediction task first: POST /api/v1/tasks with ticker='{ticker}' and interval='{interval}'"
        )
    df = db_client.get_candles(ticker, interval, from_date, to_date, limit=100000)
    if format != "csv":
        raise HTTPException(400, "Only CSV format supported currently")

    stream = io.StringIO()
    df.to_csv(stream, index=False)
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={ticker}_{interval}.csv"}
    )