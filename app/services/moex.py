# Файл: app/services/moex.py
"""
Синхронная загрузка исторических свечей с MOEX API.
Используется в отдельном потоке, чтобы не блокировать event loop.
"""
import time
import pandas as pd
import requests
import apimoex
from datetime import datetime, timedelta
from typing import Optional, Generator

from app.config import settings
from app.core.constants import CHUNK_SIZES
from app.core.exceptions import MoexAPIError
from app.core.logging import logger


def _fetch_with_retries(
    session: requests.Session,
    ticker: str,
    interval: int,
    board: str,
    start: str,
    end: str,
) -> list:
    """Выполняет запрос к MOEX с повторными попытками."""
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            return apimoex.get_board_candles(
                session=session,
                security=ticker,
                interval=interval,
                board=board,
                start=start,
                end=end,
            )
        except Exception as e:
            if attempt == max_retries:
                raise
            delay = 2 ** (attempt - 1)
            logger.warning(f"MOEX request failed (attempt {attempt}/{max_retries}): {e}. Retrying in {delay}s")
            time.sleep(delay)
    return []


def fetch_candles_sync(
    ticker: str,
    interval: str,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    board: str = "TQBR",
) -> pd.DataFrame:
    """Синхронная загрузка свечей за период."""
    if interval not in ("1m", "10m"):
        raise MoexAPIError(f"Unsupported interval: {interval}")
    interval_min = 1 if interval == "1m" else 10

    if from_date is None:
        from_date = datetime.now() - timedelta(days=14 * 365)
    if to_date is None:
        to_date = datetime.now()

    start = from_date.strftime("%Y-%m-%d")
    end = to_date.strftime("%Y-%m-%d")
    logger.debug(f"Fetching MOEX chunk {ticker} {interval} {start} -> {end}")

    with requests.Session() as session:
        session.timeout = settings.MOEX_API_TIMEOUT
        candles = _fetch_with_retries(session, ticker, interval_min, board, start, end)

    needed_columns = ["begin", "open", "high", "low", "close", "value", "volume"]
    if not candles:
        return pd.DataFrame(columns=needed_columns)

    df = pd.DataFrame(candles)
    try:
        df = df[needed_columns]
    except KeyError:
        existing = [c for c in needed_columns if c in df.columns]
        df = df[existing]
    df["begin"] = pd.to_datetime(df["begin"])
    return df


def fetch_historical_candles_stream(
    ticker: str,
    interval: str,
    years: int = 14,
) -> Generator[pd.DataFrame, None, None]:
    """Генератор, выдающий исторические свечи чанками."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    chunk_days = CHUNK_SIZES[interval]
    chunk_delta = timedelta(days=chunk_days)

    current_start = start_date
    logger.info(f"Historical download stream: {ticker} ({interval}) {start_date.date()} -> {end_date.date()}")

    while current_start < end_date:
        chunk_end = min(current_start + chunk_delta, end_date)
        try:
            df_chunk = fetch_candles_sync(ticker, interval, current_start, chunk_end)
            if not df_chunk.empty:
                df_chunk = df_chunk[["begin", "open", "high", "low", "close", "value", "volume"]]
                yield df_chunk
            else:
                logger.warning(f"Empty chunk {current_start.date()} -> {chunk_end.date()}")
        except Exception as e:
            logger.error(f"Chunk {current_start.date()} -> {chunk_end.date()} failed: {e}")
        current_start = chunk_end
        time.sleep(1.5)   # замена asyncio.sleep


def fetch_new_candles(ticker: str, interval: str, since_date: datetime) -> pd.DataFrame:
    """Докачка свечей после since_date."""
    return fetch_candles_sync(ticker, interval, from_date=since_date)