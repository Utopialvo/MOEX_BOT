# Файл: app/services/model.py
"""
Сервис для загрузки и инференса моделей CatBoost.
"""
import os
import pandas as pd
from catboost import CatBoostRegressor
from typing import Optional, Dict
from app.config import settings
from app.core.exceptions import ModelError, ERR_MODEL_NOT_FOUND
from app.core.logging import logger

FEATURE_COLUMNS = [
    "close", "high", "low", "value",
    "sma_30", "cma_30", "ema_30",
    "macd_fast", "macd_slow", "macd_signal",
    "macd", "macd_hist"
]


class ModelService:
    """Управление ML-моделями."""

    def __init__(self) -> None:
        self.models: Dict[str, CatBoostRegressor] = {}
        self._default_model: Optional[CatBoostRegressor] = None

    @property
    def default_model(self) -> CatBoostRegressor:
        """Ленивая загрузка дефолтной модели."""
        if self._default_model is None and os.path.exists(settings.DEFAULT_MODEL_PATH):
            self._default_model = self._load_model_file(settings.DEFAULT_MODEL_PATH)
        if self._default_model is None:
            raise ModelError("Default model not available")
        return self._default_model

    def _load_model_file(self, path: str) -> CatBoostRegressor:
        if not os.path.exists(path):
            raise ModelError(f"Model file not found: {path}", ERR_MODEL_NOT_FOUND)
        model = CatBoostRegressor()
        model.load_model(path)
        logger.info(f"Loaded model from {path}")
        return model

    def get_model_by_name(self, name: str) -> CatBoostRegressor:
        """Загружает модель по имени файла (без расширения) из MODEL_DIR."""
        if name in ("default", "") or not name:
            return self.default_model
        path = os.path.join(settings.MODEL_DIR, f"{name}.cbm")
        return self._load_model_file(path)

    def get_model(self, ticker: str) -> CatBoostRegressor:
        """Возвращает модель, специфичную для тикера, или дефолтную."""
        if ticker in self.models:
            return self.models[ticker]
        # Пытаемся загрузить модель тикера: {ticker}.cbm
        specific_path = os.path.join(settings.MODEL_DIR, f"{ticker}.cbm")
        if os.path.exists(specific_path):
            model = self._load_model_file(specific_path)
        else:
            model = self.default_model
        self.models[ticker] = model
        return model

    def predict_batch_with_model(self, model: CatBoostRegressor, features_df: pd.DataFrame) -> pd.Series:
        """Предсказание с использованием переданной модели."""
        if features_df.empty:
            raise ModelError("Empty features dataframe")
        X = features_df[FEATURE_COLUMNS].fillna(0)
        preds = model.predict(X)
        return pd.Series(preds, index=features_df.index)

    def predict(self, ticker: str, features_df: pd.DataFrame) -> float:
        """Предсказание для последней строки признаков."""
        if features_df.empty:
            raise ModelError("Empty features dataframe")
        last_row = features_df.iloc[-1:][FEATURE_COLUMNS].fillna(0)
        model = self.get_model(ticker)
        return float(model.predict(last_row)[0])

    def predict_batch(self, ticker: str, features_df: pd.DataFrame) -> pd.Series:
        """Пакетное предсказание с использованием модели тикера."""
        model = self.get_model(ticker)
        return self.predict_batch_with_model(model, features_df)


model_service = ModelService()