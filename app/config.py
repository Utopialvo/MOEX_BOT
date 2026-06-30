# Файл: MOEX_BOT/app/config.py
"""
Централизованные настройки приложения через переменные окружения.
Использует pydantic-settings для загрузки из .env и env vars.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Настройки MOEX_BOT."""

    # ClickHouse
    CLICKHOUSE_HOST: str = Field("localhost", env="CLICKHOUSE_HOST")
    CLICKHOUSE_PORT: int = Field(8123, env="CLICKHOUSE_PORT")
    CLICKHOUSE_USER: str = Field("default", env="CLICKHOUSE_USER")
    CLICKHOUSE_PASSWORD: str = Field("", env="CLICKHOUSE_PASSWORD")
    CLICKHOUSE_DATABASE: str = Field("default", env="CLICKHOUSE_DATABASE")

    # MOEX API
    MOEX_API_TIMEOUT: int = Field(30, env="MOEX_API_TIMEOUT")

    # Модели CatBoost
    MODEL_DIR: str = Field("/app/models", env="MODEL_DIR")
    DEFAULT_MODEL_PATH: str = Field("/app/models/reg.model", env="DEFAULT_MODEL_PATH")

    # Задачи
    MAX_CONCURRENT_TASKS: int = Field(10, env="MAX_CONCURRENT_TASKS")
    PAUSE_1M: int = Field(60, env="PAUSE_1M")
    PAUSE_10M: int = Field(600, env="PAUSE_10M")

    # Агент (read-only доступ)
    AGENT_READONLY_USER: str = Field("agent_ro", env="AGENT_READONLY_USER")
    AGENT_READONLY_PASSWORD: str = Field("", env="AGENT_READONLY_PASSWORD")

    # Логирование
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()