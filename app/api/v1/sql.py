# Файл: MOEX_BOT/app/api/v1/sql.py
"""Эндпоинт для агента: выполнение SQL-запросов только на чтение."""
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import SQLQueryRequest, SQLQueryResponse
from app.services.db import db_client
from app.core.exceptions import AgentSQLNotAllowed

router = APIRouter(prefix="/sql", tags=["SQL"])


@router.post("/query", response_model=SQLQueryResponse)
def execute_sql(request: SQLQueryRequest) -> SQLQueryResponse:
    """
    Выполняет SELECT-запрос от имени агента с ограниченными правами.
    Допускаются только запросы, начинающиеся с SELECT.
    """
    if not request.query.strip().upper().startswith("SELECT"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Only SELECT queries allowed")

    try:
        result = db_client.execute_readonly_query(request.query)
        return SQLQueryResponse(**result)
    except AgentSQLNotAllowed as e:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))