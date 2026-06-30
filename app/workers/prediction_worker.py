# Файл: app/workers/prediction_worker.py
"""
Синхронный воркер для непрерывного прогнозирования котировок.

При старте:
- Загружает указанную модель (или дефолтную для тикера).
- Загружает историю сырых свечей (чанками, с сохранением в БД).
- Полностью рассчитывает признаки и предсказания для всей истории.
- Затем переходит в регулярный цикл: докачка новых свечей,
  инкрементальный расчёт признаков и предсказаний для новых точек.

Во время ручного пересчёта истории (recalc) воркер приостанавливается
с помощью threading.Event, чтобы избежать конфликтов данных.
"""
import time
from datetime import datetime
import pandas as pd
from catboost import CatBoostRegressor

from app.config import settings
from app.services.db import db_client
from app.services.moex import fetch_new_candles, fetch_historical_candles_stream
from app.services.feature_engine import compute_incremental, compute_full
from app.services.model import model_service
from app.core.logging import logger
from app.core.constants import MIN_VALID_DATE


def run_prediction_loop(
    ticker: str,
    interval: str,
    task_id: str,
    task_manager: 'TaskManager',
    model_name: str = None,
) -> None:
    """
    Основной цикл прогнозирования. Все операции синхронные,
    так как воркер запускается в отдельном потоке.
    """
    pause = settings.PAUSE_1M if interval == "1m" else settings.PAUSE_10M
    logger.info(f"Worker started: {ticker}_{interval} (task_id={task_id}, model={model_name})")

    # ------------------ Загрузка модели ------------------
    if model_name:
        try:
            model = model_service.get_model_by_name(model_name)
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise
    else:
        model = model_service.get_model(ticker)
    logger.info(f"Model loaded for {ticker}: {model_name or 'default'}")

    # ------------------------------------------------
    # Этап 1: Первоначальная загрузка истории
    # ------------------------------------------------
    last_date = db_client.get_last_candle_date(ticker, interval)
    min_valid = pd.Timestamp(MIN_VALID_DATE)

    if last_date is None or last_date < min_valid:
        logger.info(f"Starting historical download for {ticker}_{interval}")
        for chunk_df in fetch_historical_candles_stream(ticker, interval):
            if not chunk_df.empty:
                db_client.insert_raw_candles(ticker, interval, chunk_df)
                logger.info(f"Inserted chunk of {len(chunk_df)} candles")
        time.sleep(2)   # небольшая пауза для завершения фоновых слияний
        last_date = db_client.get_last_candle_date(ticker, interval)

    if last_date is None:
        logger.error(f"Failed to load any data for {ticker}_{interval}. Exiting.")
        return

    # ------------------------------------------------
    # Этап 2: Полный пересчёт признаков и предсказаний
    # ------------------------------------------------
    logger.info("Performing full feature & prediction computation for history")
    try:
        features_df = compute_full(ticker, interval)
        if not features_df.empty:
            db_client.insert_features(ticker, interval, features_df)
            preds = model_service.predict_batch_with_model(model, features_df)
            records = []
            for idx, row in features_df.iterrows():
                records.append({
                    "timestamp": row["begin"],
                    "value": preds[idx],
                    "ticker": ticker,
                    "interval": interval,
                    "is_history": 1,
                })
            db_client.insert_predictions_batch(ticker, interval, records)
            logger.info(f"History predictions inserted: {len(records)}")
    except Exception as e:
        logger.error(f"Full recomputation failed: {e}")

    # ------------------------------------------------
    # Этап 3: Регулярный цикл обновления
    # ------------------------------------------------
    consecutive_errors = 0
    while True:
        # Проверка: не находится ли задача в режиме паузы (для ручного пересчёта)
        pause_event = task_manager._pause_events.get(task_id)
        if pause_event and pause_event.is_set():
            time.sleep(1)
            continue

        try:
            last_date = db_client.get_last_candle_date(ticker, interval)
            if last_date is None:
                time.sleep(60)
                continue

            # Докачка новых свечей
            new_df = fetch_new_candles(ticker, interval, last_date)
            if not new_df.empty:
                new_df = new_df[["begin", "open", "high", "low", "close", "volume", "value"]]
                db_client.insert_raw_candles(ticker, interval, new_df)
                logger.info(f"Inserted {len(new_df)} new candles")

                # Инкрементальный расчёт признаков
                new_features = compute_incremental(ticker, interval, new_df)
                if not new_features.empty:
                    db_client.insert_features(ticker, interval, new_features)

                    # Предсказания для новых точек
                    new_preds = model_service.predict_batch_with_model(model, new_features)
                    records = []
                    for idx, row in new_features.iterrows():
                        records.append({
                            "timestamp": row["begin"],
                            "value": new_preds[idx],
                            "ticker": ticker,
                            "interval": interval,
                            "is_history": 0,
                        })
                    db_client.insert_predictions_batch(ticker, interval, records)
                    logger.info(f"Inserted {len(records)} new predictions")

            # Обновляем статус задачи
            if task_id in task_manager.tasks:
                task_manager.tasks[task_id]["last_run"] = datetime.now()
                task_manager.tasks[task_id]["status"] = "running"

            consecutive_errors = 0
            time.sleep(pause)

        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Error in worker {ticker}_{interval}: {e}", exc_info=True)
            if task_id in task_manager.tasks:
                task_manager.tasks[task_id]["status"] = "error"
                task_manager.tasks[task_id]["error_message"] = str(e)
            if consecutive_errors >= 10:
                logger.critical(f"Too many errors, stopping worker {ticker}_{interval}")
                if task_id in task_manager.tasks:
                    task_manager.tasks[task_id]["status"] = "stopped"
                break
            time.sleep(60)