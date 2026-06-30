# Файл: MOEX_BOT/app/core/exceptions.py
"""Кастомные исключения с кодами ошибок и глобальный обработчик."""
from fastapi import Request, status
from fastapi.responses import JSONResponse


class MoexBotError(Exception):
    """Базовое исключение приложения с кодом ошибки."""
    def __init__(self, message: str, error_code: int = 1000):
        super().__init__(message)
        self.error_code = error_code


# Коды ошибок (для удобства)
ERR_TASK_NOT_FOUND = 1001
ERR_INVALID_INTERVAL = 1002
ERR_MAX_TASKS = 1003
ERR_TABLE_NOT_FOUND = 2001
ERR_CLICKHOUSE = 2002
ERR_MODEL_NOT_FOUND = 3001
ERR_MODEL_LOAD = 3002
ERR_MOEX_API = 4001
ERR_SQL_NOT_ALLOWED = 5001


class ClickHouseError(MoexBotError):
    def __init__(self, message: str):
        super().__init__(message, ERR_CLICKHOUSE)


class MoexAPIError(MoexBotError):
    def __init__(self, message: str):
        super().__init__(message, ERR_MOEX_API)


class ModelError(MoexBotError):
    def __init__(self, message: str, code: int = ERR_MODEL_LOAD):
        super().__init__(message, code)


class TaskError(MoexBotError):
    def __init__(self, message: str, code: int = 1000):
        super().__init__(message, code)


class AgentSQLNotAllowed(MoexBotError):
    def __init__(self, message: str = "Only SELECT queries allowed"):
        super().__init__(message, ERR_SQL_NOT_ALLOWED)


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Глобальный обработчик, возвращающий код ошибки."""
    if isinstance(exc, MoexBotError):
        status_code = status.HTTP_400_BAD_REQUEST
        detail = str(exc)
        error_code = exc.error_code
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        detail = "Internal server error"
        error_code = 9999
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "type": exc.__class__.__name__, "error_code": error_code},
    )