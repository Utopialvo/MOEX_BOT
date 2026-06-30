# Файл: app/workers/task_manager.py
"""
Менеджер фоновых задач прогнозирования.
С автоматическим перезапуском при падениях (кроме штатной остановки).
"""
import asyncio
import uuid
import threading
from datetime import datetime
from typing import Dict, Optional, List

from app.config import settings
from app.core.exceptions import TaskError, ERR_TASK_NOT_FOUND
from app.core.logging import logger
from app.services.model import model_service
from app.services.db import db_client
from app.services.feature_engine import compute_full
from app.workers.prediction_worker import run_prediction_loop


class TaskManager:
    """Хранит и контролирует жизненный цикл воркеров."""

    def __init__(self) -> None:
        self.tasks: Dict[str, Dict] = {}
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)
        self._lock = asyncio.Lock()
        self._pause_events: Dict[str, threading.Event] = {}   # синхронные события
        self._stop_events: Dict[str, threading.Event] = {}    # флаги штатной остановки

    async def start_task(self, ticker: str, interval: str, model_name: Optional[str] = None) -> str:
        async with self._lock:
            if len(self.tasks) >= settings.MAX_CONCURRENT_TASKS:
                raise TaskError("Max concurrent tasks reached", code=1003)

            db_client.ensure_tables_for_ticker(ticker, interval)

            task_id = str(uuid.uuid4())
            self.tasks[task_id] = {
                "ticker": ticker,
                "interval": interval,
                "status": "pending",
                "created_at": datetime.now(),
                "last_run": None,
                "future": None,
                "error_message": None,
                "model_name": model_name,
                "recalc_status": "idle",
                "stopped_by_user": False,
            }
            self._pause_events[task_id] = threading.Event()
            self._stop_events[task_id] = threading.Event()

        await self._semaphore.acquire()
        try:
            # Запускаем воркер в отдельном потоке, оборачивая в asyncio.to_thread
            task_future = asyncio.ensure_future(self._run_worker(task_id, ticker, interval, model_name))
            async with self._lock:
                self.tasks[task_id]["future"] = task_future
                self.tasks[task_id]["status"] = "running"
            logger.info(f"Task {task_id} started ({ticker}_{interval}) model={model_name}")
            return task_id
        except Exception as e:
            self._semaphore.release()
            async with self._lock:
                if task_id in self.tasks:
                    self.tasks[task_id]["status"] = "error"
                    self.tasks[task_id]["error_message"] = str(e)
                self._pause_events.pop(task_id, None)
                self._stop_events.pop(task_id, None)
            raise TaskError(f"Failed to start task: {e}")

    async def _run_worker(self, task_id: str, ticker: str, interval: str, model_name: Optional[str]) -> None:
        """
        Цикл, управляющий жизнью воркера. После завершения (штатного или аварийного)
        проверяет флаг stopped_by_user и, если надо, перезапускает с задержкой.
        """
        restart_delay = 10   # начальная задержка для автоматического рестарта
        max_restart_delay = 600
        while True:
            stop_event = self._stop_events.get(task_id)
            if stop_event and stop_event.is_set():
                break

            try:
                # Запускаем синхронный воркер в потоке
                await asyncio.to_thread(
                    run_prediction_loop,
                    ticker, interval, task_id, self, model_name
                )
            except asyncio.CancelledError:
                # Отмена задачи (например, при остановке сервера) – выходим без рестарта
                break
            except Exception as e:
                logger.error(f"Worker thread for {ticker}_{interval} crashed: {e}")

            # После выхода из run_prediction_loop (штатного или из-за ошибок) проверяем, нужен ли рестарт
            async with self._lock:
                info = self.tasks.get(task_id)
                if not info:
                    break
                if info.get("stopped_by_user"):
                    break
                # Экспоненциальная задержка перед рестартом
                logger.info(f"Restarting task {task_id} in {restart_delay}s")
                info["status"] = "pending"
            await asyncio.sleep(restart_delay)
            restart_delay = min(restart_delay * 2, max_restart_delay)

        # Финализация: освобождаем семафор и чистим структуры
        self._semaphore.release()
        async with self._lock:
            if task_id in self.tasks:
                if not self.tasks[task_id].get("stopped_by_user"):
                    self.tasks[task_id]["status"] = "stopped"
                self._pause_events.pop(task_id, None)
                self._stop_events.pop(task_id, None)
        logger.info(f"Task {task_id} fully finished")

    async def stop_task(self, task_id: str) -> None:
        """Штатная остановка задачи с пометкой stopped_by_user."""
        async with self._lock:
            info = self.tasks.get(task_id)
            if not info:
                raise TaskError("Task not found", ERR_TASK_NOT_FOUND)
            info["stopped_by_user"] = True
            # Устанавливаем стоп-событие
            stop_event = self._stop_events.get(task_id)
            if stop_event:
                stop_event.set()
            future = info.get("future")
            self._pause_events.pop(task_id, None)
        if future and not future.done():
            future.cancel()
            try:
                await future
            except asyncio.CancelledError:
                pass

    async def restart_task(self, task_id: str) -> str:
        """Принудительный перезапуск с новым ID."""
        async with self._lock:
            info = self.tasks.get(task_id)
            if not info:
                raise TaskError("Task not found", ERR_TASK_NOT_FOUND)
            ticker, interval, model_name = info["ticker"], info["interval"], info.get("model_name")
        await self.stop_task(task_id)
        # Ждём освобождения ресурсов
        await asyncio.sleep(1)
        async with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
        return await self.start_task(ticker, interval, model_name)

    async def recalc_history_async(self, task_id: str) -> None:
        """Запускает асинхронный пересчёт истории и сразу возвращает управление."""
        async with self._lock:
            info = self.tasks.get(task_id)
            if not info:
                raise TaskError("Task not found", ERR_TASK_NOT_FOUND)
            if info["recalc_status"] == "in_progress":
                raise TaskError("Recalculation already in progress")
            info["recalc_status"] = "in_progress"
            ticker = info["ticker"]
            interval = info["interval"]
            pause_event = self._pause_events.get(task_id)

        # Запускаем фоновый пересчёт в том же цикле событий, но без блокировки
        asyncio.create_task(self._run_recalc(task_id, ticker, interval, pause_event))

    async def _run_recalc(self, task_id: str, ticker: str, interval: str,
                          pause_event: Optional[threading.Event]) -> None:
        """Фактическое выполнение пересчёта в фоне."""
        try:
            if pause_event:
                pause_event.set()          # паузим воркер
                await asyncio.sleep(2)     # даём время завершить текущую итерацию

            logger.info(f"Starting async full history recalc for {ticker}_{interval}")
            # Выполняем compute_full (CPU-bound) в потоке
            features_df = await asyncio.to_thread(compute_full, ticker, interval)
            if not features_df.empty:
                # Вставка признаков (синхронная) тоже в поток, чтобы не блокировать
                await asyncio.to_thread(db_client.insert_features, ticker, interval, features_df)

                # Получаем модель задачи
                model = await asyncio.to_thread(self._get_model_for_task, task_id)
                preds = await asyncio.to_thread(
                    model_service.predict_batch_with_model, model, features_df
                )
                records = []
                for idx, row in features_df.iterrows():
                    records.append({
                        "timestamp": row["begin"],
                        "value": preds[idx],
                        "ticker": ticker,
                        "interval": interval,
                        "is_history": 1,
                    })
                await asyncio.to_thread(db_client.insert_predictions_batch, ticker, interval, records)
                logger.info(f"Recalc complete: {len(records)} predictions")

            async with self._lock:
                if task_id in self.tasks:
                    self.tasks[task_id]["recalc_status"] = "completed"
        except Exception as e:
            logger.error(f"History recalc failed: {e}")
            async with self._lock:
                if task_id in self.tasks:
                    self.tasks[task_id]["recalc_status"] = "failed"
        finally:
            if pause_event:
                pause_event.clear()  # снимаем паузу

    def _get_model_for_task(self, task_id: str) -> 'CatBoostRegressor':
        """Возвращает модель, ассоциированную с задачей, или дефолтную."""
        info = self.tasks.get(task_id)
        if info and info.get("model_name"):
            return model_service.get_model_by_name(info["model_name"])
        return model_service.default_model or model_service.get_model(info["ticker"])

    def get_status(self, task_id: str) -> Optional[Dict]:
        info = self.tasks.get(task_id)
        return info.copy() if info else None

    def list_tasks(self, status_filter: Optional[str] = None) -> List[Dict]:
        result = []
        for tid, info in self.tasks.items():
            if status_filter and info["status"] != status_filter:
                continue
            result.append({"task_id": tid, **info})
        return result


task_manager = TaskManager()