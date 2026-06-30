# Файл: app/api/v1/tasks.py
"""
Управление задачами прогнозирования.
"""
from fastapi import APIRouter, HTTPException, status
from uuid import UUID
from app.models.schemas import TaskCreate, BulkTaskCreate, TaskInfo, TaskListResponse
from app.workers.task_manager import task_manager
from app.core.exceptions import TaskError, ERR_TASK_NOT_FOUND

router = APIRouter(prefix="/tasks", tags=["Tasks"])


def _to_task_info(task_id: str, info: dict) -> TaskInfo:
    return TaskInfo(
        task_id=UUID(task_id),
        ticker=info["ticker"],
        interval=info["interval"],
        status=info["status"],
        created_at=info["created_at"],
        last_run=info.get("last_run"),
        error_message=info.get("error_message"),
        model_name=info.get("model_name"),
        recalc_status=info.get("recalc_status", "idle"),
    )


@router.post("", response_model=TaskInfo, status_code=201)
async def create_task(task_data: TaskCreate) -> TaskInfo:
    """Запуск новой задачи прогнозирования."""
    try:
        task_id = await task_manager.start_task(
            ticker=task_data.ticker,
            interval=task_data.interval,
            model_name=task_data.model_name,
        )
        info = task_manager.get_status(task_id)
        return _to_task_info(task_id, info)
    except TaskError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/bulk", status_code=201)
async def bulk_create_tasks(bulk: BulkTaskCreate):
    """Массовый запуск задач."""
    results = []
    errors = []
    for task_data in bulk.tasks:
        try:
            task_id = await task_manager.start_task(
                ticker=task_data.ticker,
                interval=task_data.interval,
                model_name=task_data.model_name,
            )
            info = task_manager.get_status(task_id)
            results.append(_to_task_info(task_id, info))
        except TaskError as e:
            errors.append({
                "ticker": task_data.ticker,
                "interval": task_data.interval,
                "error": str(e)
            })
    return {"created": [t.dict() for t in results], "errors": errors}


@router.get("", response_model=TaskListResponse)
async def list_tasks(status: str | None = None) -> TaskListResponse:
    """Получить список всех задач с опциональным фильтром по статусу."""
    tasks = task_manager.list_tasks(status)
    items = [_to_task_info(t["task_id"], t) for t in tasks]
    return TaskListResponse(tasks=items, total=len(items))


@router.get("/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str) -> TaskInfo:
    """Детальная информация о задаче."""
    info = task_manager.get_status(task_id)
    if not info:
        raise HTTPException(status_code=404, detail="Task not found")
    return _to_task_info(task_id, info)


@router.delete("/{task_id}", status_code=204)
async def stop_task(task_id: str) -> None:
    """Штатная остановка задачи."""
    try:
        await task_manager.stop_task(task_id)
    except TaskError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{task_id}/restart", response_model=TaskInfo)
async def restart_task(task_id: str) -> TaskInfo:
    """Принудительный перезапуск задачи."""
    try:
        new_id = await task_manager.restart_task(task_id)
        info = task_manager.get_status(new_id)
        return _to_task_info(new_id, info)
    except TaskError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{task_id}/recalc-predictions", status_code=202)
async def recalc_predictions(task_id: str):
    """Запуск фонового пересчёта истории предсказаний."""
    try:
        await task_manager.recalc_history_async(task_id)
        return {"message": "Recalculation started", "task_id": task_id}
    except TaskError as e:
        raise HTTPException(status_code=400, detail=str(e))