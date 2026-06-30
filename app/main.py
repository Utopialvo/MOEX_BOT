# Файл: MOEX_BOT/app/main.py
"""
Главный модуль FastAPI приложения MOEX_BOT.
Собирает все роутеры и запускает сервисы.
"""
from fastapi import FastAPI
from app.api.v1 import tasks, data, system, sql, monitoring, models, export
from app.core.exceptions import MoexBotError, global_exception_handler
from app.core.logging import logger
from app.config import settings

app = FastAPI(
    title="MOEX_BOT API",
    description="Система прогнозирования котировок Мосбиржи с поддержкой агентов",
    version="1.2.0",
)

# Роутеры
app.include_router(tasks.router, prefix="/api/v1")
app.include_router(data.router, prefix="/api/v1")
app.include_router(system.router, prefix="/api/v1")
app.include_router(sql.router, prefix="/api/v1")
app.include_router(monitoring.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(export.router, prefix="/api/v1")

# Глобальный обработчик ошибок
app.add_exception_handler(MoexBotError, global_exception_handler)


@app.on_event("startup")
async def startup() -> None:
    """Проверка ключевых сервисов при старте."""
    logger.info("MOEX_BOT starting up")
    from app.services.db import db_client
    try:
        # Пинг ClickHouse
        db_client._get_client().ping()
        logger.info("ClickHouse connection OK")
    except Exception as e:
        logger.error(f"ClickHouse connection failed: {e}")

    # Проверяем доступность дефолтной модели (ленивая загрузка произойдёт позже)
    from app.services.model import model_service
    try:
        _ = model_service.default_model   # загрузит, если есть файл
        logger.info("Default model loaded OK")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")


@app.on_event("shutdown")
async def shutdown() -> None:
    """Останавливаем все задачи перед выключением."""
    logger.info("Shutting down, cancelling tasks...")
    from app.workers.task_manager import task_manager
    for task_id in list(task_manager.tasks.keys()):
        await task_manager.stop_task(task_id)
    logger.info("All tasks stopped")