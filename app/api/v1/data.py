# Файл: app/api/v1/data.py
"""
Эндпоинты для получения рыночных данных: свечи, предсказания, агрегаты,
сигналы, корреляции, а также batch-запросы для агентов.
Все обработчики синхронные, FastAPI выполняет их в пуле потоков.
"""
import pandas as pd
from fastapi import APIRouter, Query, HTTPException
from datetime import datetime
from typing import List, Optional
from app.services.db import db_client
from app.models.schemas import (
    CandleListResponse,
    CandleResponse,
    PredictionResponse,
    PredictionHistoryResponse,
    AggregateResponse,
    SignalResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchCandleRequest,
    BatchCandleResponse,
)
from app.core.constants import (
    SUPPORTED_INTERVALS,
    get_raw_candles_table,
    get_predictions_table,
    get_hourly_agg_table,
    get_daily_agg_table,
    get_features_table,
)

router = APIRouter(prefix="/data", tags=["Data"])


def _check_table_or_404(table_name: str, ticker: str, interval: str) -> None:
    if not db_client.table_exists(table_name):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Таблица/представление '{table_name}' не найдена. "
                f"Сначала запустите задачу: POST /api/v1/tasks с ticker='{ticker}' и interval='{interval}'"
            ),
        )


@router.get("/candles/{ticker}", response_model=CandleListResponse)
def get_candles(
    ticker: str,
    interval: str = Query(..., description="Интервал (1m, 10m)"),
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    limit: int = Query(1000, ge=1, le=10000),
) -> CandleListResponse:
    """Получить исторические свечи."""
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(400, "Неподдерживаемый интервал")
    table = get_raw_candles_table(ticker, interval)
    _check_table_or_404(table, ticker, interval)

    try:
        df = db_client.get_candles(ticker, interval, from_date, to_date, limit)
        candles = [
            CandleResponse(
                begin=row["begin"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                value=float(row["value"]),
                volume=int(row["volume"]),
            )
            for _, row in df.iterrows()
        ]
        return CandleListResponse(candles=candles, ticker=ticker, interval=interval)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/predictions/{ticker}", response_model=PredictionResponse)
def get_last_prediction(
    ticker: str,
    interval: str = Query(..., description="Интервал (1m, 10m)"),
) -> PredictionResponse:
    """Получить последнее предсказание."""
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(400, "Неподдерживаемый интервал")
    table = get_predictions_table(ticker, interval)
    _check_table_or_404(table, ticker, interval)

    df = db_client.get_predictions(ticker, interval, limit=1)
    if df.empty:
        raise HTTPException(404, "Предсказания отсутствуют")
    row = df.iloc[0]
    return PredictionResponse(
        timestamp=row["timestamp"],
        value=float(row["value"]),
        ticker=row["ticker"],
        interval=row["interval"],
    )


@router.get("/predictions/{ticker}/history", response_model=PredictionHistoryResponse)
def get_prediction_history(
    ticker: str,
    interval: str = Query(..., description="Интервал (1m, 10m)"),
    limit: int = 100,
) -> PredictionHistoryResponse:
    """История предсказаний."""
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(400, "Неподдерживаемый интервал")
    table = get_predictions_table(ticker, interval)
    _check_table_or_404(table, ticker, interval)

    df = db_client.get_predictions(ticker, interval, limit)
    predictions = [
        PredictionResponse(
            timestamp=row["timestamp"],
            value=float(row["value"]),
            ticker=row["ticker"],
            interval=row["interval"],
        )
        for _, row in df.iterrows()
    ]
    return PredictionHistoryResponse(predictions=predictions)


@router.get("/aggregates/{ticker}", response_model=List[AggregateResponse])
def get_aggregates(
    ticker: str,
    interval: str = Query(..., description="Интервал (1m, 10m)"),
    granularity: str = Query("hourly", description="hourly или daily"),
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=1000),
) -> List[AggregateResponse]:
    """Агрегированные данные (по часам или дням)."""
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(400, "Неподдерживаемый интервал")
    if granularity not in ("hourly", "daily"):
        raise HTTPException(400, "Гранулярность должна быть 'hourly' или 'daily'")

    if granularity == "hourly":
        table = get_hourly_agg_table(ticker, interval)
        time_col = "hour"
    else:
        table = get_daily_agg_table(ticker, interval)
        time_col = "day"

    _check_table_or_404(table, ticker, interval)

    conditions = ["1=1"]
    if from_date:
        conditions.append(f"{time_col} >= '{from_date}'")
    if to_date:
        conditions.append(f"{time_col} <= '{to_date}'")
    where = " AND ".join(conditions)

    query = f"""
        SELECT
            {time_col},
            avgMerge(close_avg) AS avg_price,
            maxMerge(high_max) AS max_price,
            minMerge(low_min) AS min_price,
            sumMerge(volume_sum) AS total_volume,
            countMerge(trade_count) AS count
        FROM {table}
        WHERE {where}
        GROUP BY {time_col}
        ORDER BY {time_col} DESC
        LIMIT {limit}
    """
    try:
        df = db_client._get_client().query_df(query)
    except Exception as e:
        raise HTTPException(500, f"Ошибка выполнения запроса: {e}")

    return [
        AggregateResponse(
            timestamp=row[time_col],
            avg_price=float(row["avg_price"]),
            max_price=float(row["max_price"]),
            min_price=float(row["min_price"]),
            total_volume=int(row["total_volume"]),
            count=int(row["count"]),
        )
        for _, row in df.iterrows()
    ]


@router.get("/signals/{ticker}", response_model=List[SignalResponse])
def get_signals(
    ticker: str,
    interval: str = Query(..., description="Интервал (1m, 10m)"),
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=1000),
) -> List[SignalResponse]:
    """Торговые сигналы на пересечении SMA-12 и SMA-26."""
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(400, "Неподдерживаемый интервал")

    features_table = get_features_table(ticker, interval)
    _check_table_or_404(features_table, ticker, interval)
    
    query = f"""
        SELECT
            begin,
            buy_signal,
            sell_signal
        FROM (
            SELECT
                begin,
                sma_12,
                sma_26,
                if(
                    sma_12 > sma_26
                    AND lagInFrame(sma_12, 1) OVER (ORDER BY begin)
                        <= lagInFrame(sma_26, 1) OVER (ORDER BY begin),
                    1, 0
                ) AS buy_signal,
                if(
                    sma_12 < sma_26
                    AND lagInFrame(sma_12, 1) OVER (ORDER BY begin)
                        >= lagInFrame(sma_26, 1) OVER (ORDER BY begin),
                    1, 0
                ) AS sell_signal
            FROM (
                SELECT
                    begin,
                    avg(close) OVER (
                        ORDER BY begin ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                    ) AS sma_12,
                    avg(close) OVER (
                        ORDER BY begin ROWS BETWEEN 25 PRECEDING AND CURRENT ROW
                    ) AS sma_26
                FROM {features_table}
                WHERE 1=1
                {'AND begin >= \'' + from_date.strftime('%Y-%m-%d %H:%M:%S') + '\'' if from_date else ''}
                {'AND begin <= \'' + to_date.strftime('%Y-%m-%d %H:%M:%S') + '\'' if to_date else ''}
            )
        )
        ORDER BY begin DESC
        LIMIT {limit}
    """
    try:
        df = db_client._get_client().query_df(query)
    except Exception as e:
        raise HTTPException(500, f"Ошибка выполнения запроса: {e}")

    return [
        SignalResponse(
            begin=row["begin"],
            buy_signal=int(row["buy_signal"]),
            sell_signal=int(row["sell_signal"]),
        )
        for _, row in df.iterrows()
    ]


@router.get("/correlations")
def get_correlations(
    tickers: str = Query(..., description="Тикеры через запятую (минимум 2)"),
    interval: str = Query(..., description="Интервал (1m, 10m)"),
    window: int = Query(100, ge=10, le=1000, description="Количество последних свечей для расчёта"),
) -> dict:
    """Корреляции между тикерами на основе последних свечей."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(ticker_list) < 2:
        raise HTTPException(400, "Укажите минимум два тикера через запятую")
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(400, "Неподдерживаемый интервал")

    for t in ticker_list:
        table = get_raw_candles_table(t, interval)
        if not db_client.table_exists(table):
            raise HTTPException(404, f"Таблица {table} не найдена. Сначала запустите задачу для {t}.")

    try:
        data = {}
        common_begins = None
        for t in ticker_list:
            df = db_client.get_candles(t, interval, limit=window)
            if df.empty:
                raise HTTPException(404, f"Нет данных для {t}")
            df = df[["begin", "close"]].copy()
            df.set_index("begin", inplace=True)
            data[t] = df["close"]
            if common_begins is None:
                common_begins = set(df.index)
            else:
                common_begins &= set(df.index)

        if len(common_begins) < 10:
            raise HTTPException(400, "Недостаточно общих временных меток для расчёта корреляции")

        common_begins = sorted(common_begins)[-window:]
        aligned = pd.DataFrame(index=common_begins)
        for t in ticker_list:
            aligned[t] = data[t].reindex(common_begins)

        corr_matrix = aligned.corr()
        result = []
        for i in range(len(ticker_list)):
            for j in range(i + 1, len(ticker_list)):
                t1, t2 = ticker_list[i], ticker_list[j]
                result.append({
                    "ticker1": t1,
                    "ticker2": t2,
                    "correlation": round(float(corr_matrix.loc[t1, t2]), 4),
                })

        return {
            "interval": interval,
            "window": len(common_begins),
            "tickers": ticker_list,
            "correlations": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Ошибка при вычислении корреляций: {e}")


@router.post("/predictions/batch", response_model=BatchPredictionResponse)
def batch_predictions(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Получить последние предсказания для нескольких тикеров одним запросом."""
    if request.interval not in SUPPORTED_INTERVALS:
        raise HTTPException(400, "Неподдерживаемый интервал")
    result = {}
    for ticker in request.tickers:
        table = get_predictions_table(ticker, request.interval)
        if not db_client.table_exists(table):
            result[ticker] = None
            continue
        df = db_client.get_predictions(ticker, request.interval, limit=1)
        if df.empty:
            result[ticker] = None
        else:
            row = df.iloc[0]
            result[ticker] = PredictionResponse(
                timestamp=row["timestamp"],
                value=float(row["value"]),
                ticker=row["ticker"],
                interval=row["interval"],
            )
    return BatchPredictionResponse(predictions=result)


@router.post("/candles/batch", response_model=BatchCandleResponse)
def batch_candles(request: BatchCandleRequest) -> BatchCandleResponse:
    """Получить свечи для нескольких тикеров."""
    if request.interval not in SUPPORTED_INTERVALS:
        raise HTTPException(400, "Неподдерживаемый интервал")
    result = {}
    for ticker in request.tickers:
        table = get_raw_candles_table(ticker, request.interval)
        if not db_client.table_exists(table):
            result[ticker] = []
            continue
        df = db_client.get_candles(ticker, request.interval, limit=request.limit)
        candles = [
            CandleResponse(
                begin=row["begin"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                value=float(row["value"]),
                volume=int(row["volume"]),
            )
            for _, row in df.iterrows()
        ]
        result[ticker] = candles
    return BatchCandleResponse(candles=result)