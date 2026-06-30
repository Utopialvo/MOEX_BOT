# Файл: app/core/constants.py
"""
Константы и вспомогательные функции для формирования имён таблиц и представлений.
"""
SUPPORTED_INTERVALS = {"1m": 1, "10m": 10}

# Размеры чанков для загрузки исторических данных (не более 5000 свечей)
CHUNK_SIZES = {
    "1m": 7,    # дней
    "10m": 70,  # дней
}

MIN_VALID_DATE = "2000-01-01"


def get_raw_candles_table(ticker: str, interval: str) -> str:
    """Сырые свечи (ReplacingMergeTree)."""
    return f"raw_candles_{ticker}_{interval}"

def get_features_table(ticker: str, interval: str) -> str:
    """Признаки для модели (MergeTree)."""
    return f"features_{ticker}_{interval}"

def get_predictions_table(ticker: str, interval: str) -> str:
    """Предсказания (MergeTree)."""
    return f"predictions_{ticker}_{interval}"

def get_hourly_agg_table(ticker: str, interval: str) -> str:
    """Почасовые агрегаты (AggregatingMergeTree)."""
    return f"aggr_hourly_{ticker}_{interval}"

def get_daily_agg_table(ticker: str, interval: str) -> str:
    """Подневные агрегаты (AggregatingMergeTree)."""
    return f"aggr_daily_{ticker}_{interval}"