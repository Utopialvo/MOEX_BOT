# Файл: app/services/feature_engine.py
"""
Инкрементальный и полный расчёт признаков.
"""
import pandas as pd
from typing import Optional
from app.core.logging import logger
from app.services.db import db_client


WINDOW_SMA = 30
WINDOW_MACD_FAST = 12
WINDOW_MACD_SLOW = 26
WINDOW_MACD_SIGNAL = 9
EMA_ALPHA = 2 / (WINDOW_SMA + 1)
# Количество строк для чанковой обработки в compute_full
FULL_BATCH_SIZE = 100_000


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет все признаки:
    sma_30, cma_30, ema_30, macd_fast, macd_slow, macd_signal, macd, macd_hist.
    Возвращает DataFrame с колонками признаков.
    """
    df = df.sort_values("begin").copy()
    df["sma_30"] = df["close"].rolling(window=WINDOW_SMA, min_periods=1).mean()
    df["cma_30"] = df["sma_30"] * 0.967 + df["close"] * 0.033
    df["ema_30"] = df["close"].ewm(alpha=EMA_ALPHA, adjust=False).mean()
    df["macd_fast"] = df["close"].ewm(span=WINDOW_MACD_FAST, adjust=False).mean()
    df["macd_slow"] = df["close"].ewm(span=WINDOW_MACD_SLOW, adjust=False).mean()
    df["macd"] = df["macd_fast"] - df["macd_slow"]
    df["macd_signal"] = df["macd"].ewm(span=WINDOW_MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df[[
        "begin", "close", "high", "low", "value",
        "sma_30", "cma_30", "ema_30",
        "macd_fast", "macd_slow", "macd_signal",
        "macd", "macd_hist"
    ]]


def compute_incremental(
    ticker: str,
    interval: str,
    new_candles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Инкрементальный расчёт признаков для новых свечей.
    Использует последние WINDOW_SMA строк контекста из таблицы признаков.
    """
    context = db_client.get_last_features_context(ticker, interval, size=WINDOW_SMA)
    if context.empty or "begin" not in context.columns:
        logger.info(f"No previous features for {ticker}_{interval}, starting from scratch")
        return compute_full(ticker, interval, start_date=new_candles["begin"].min())

    combined = pd.concat([context, new_candles], ignore_index=True)
    combined.drop_duplicates(subset="begin", keep="last", inplace=True)
    combined.sort_values("begin", inplace=True)

    feats = _compute_features(combined)

    new_begins = set(new_candles["begin"])
    new_features = feats[feats["begin"].isin(new_begins)]

    logger.info(f"Computed {len(new_features)} incremental features for {ticker}_{interval}")
    return new_features


def compute_full(
    ticker: str,
    interval: str,
    start_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Полный расчёт признаков по всем сырым свечам чанками.
    Загружает данные из ClickHouse порциями, чтобы не перегружать память.
    При перекрытии чанков обеспечивается корректность скользящих окон.
    """
    raw_table = f"raw_candles_{ticker}_{interval}"
    base_query = f"SELECT * FROM {raw_table}"
    if start_date:
        base_query += f" WHERE begin >= '{start_date}'"
    base_query += " ORDER BY begin"

    offset = 0
    all_features = []
    prev_tail = pd.DataFrame()  # хвост предыдущего чанка для сохранения контекста

    while True:
        query = base_query + f" LIMIT {FULL_BATCH_SIZE} OFFSET {offset}"
        chunk = db_client._get_client().query_df(query)
        if chunk.empty:
            break

        # Если есть предыдущий хвост, склеиваем для обеспечения непрерывности расчёта
        if not prev_tail.empty:
            chunk = pd.concat([prev_tail, chunk], ignore_index=True)

        # Убираем дубликаты, которые могли возникнуть из-за перекрытия
        chunk.drop_duplicates(subset="begin", keep="last", inplace=True)
        chunk.sort_values("begin", inplace=True)

        # Вычисляем признаки для склеенного чанка
        features = _compute_features(chunk)

        # Отделяем ту часть, которая относится к текущему чанку (без хвоста)
        if not prev_tail.empty:
            # prev_tail мог быть меньше WINDOW_SMA, но мы оставляем только новые строки
            new_begins = set(chunk["begin"]) - set(prev_tail["begin"])
            features = features[features["begin"].isin(new_begins)]

        all_features.append(features)

        # Оставляем хвост из последних WINDOW_SMA строк для следующего чанка
        if len(chunk) >= WINDOW_SMA:
            prev_tail = chunk.iloc[-WINDOW_SMA:][["begin", "open", "high", "low", "close", "volume", "value"]]
        else:
            prev_tail = chunk.copy()

        offset += FULL_BATCH_SIZE

    if not all_features:
        logger.warning(f"No raw candles for {ticker}_{interval}")
        return pd.DataFrame()

    result = pd.concat(all_features, ignore_index=True)
    logger.info(f"Full feature computation for {ticker}_{interval}: {len(result)} rows")
    return result