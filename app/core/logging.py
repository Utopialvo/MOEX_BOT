# Файл: MOEX_BOT/app/core/logging.py
"""Настройка логгера."""
import logging
import sys
from app.config import settings


def setup_logging() -> logging.Logger:
    """Инициализирует корневой логгер с заданным уровнем."""
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Понижаем уровень для шумных библиотек
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("clickhouse_connect").setLevel(logging.WARNING)
    return logging.getLogger("moex_bot")


logger = setup_logging()