# Файл: app/services/db.py
"""
Клиент для взаимодействия с ClickHouse.
Обеспечивает создание таблиц, вставку и чтение данных,
а также read-only запросы для агентов.
"""
import pandas as pd
import clickhouse_connect
from datetime import datetime
from typing import Dict, Any, Optional, List

from app.config import settings
from app.core.constants import *
from app.core.exceptions import ClickHouseError, AgentSQLNotAllowed
from app.core.logging import logger

BATCH_SIZE = 1_000


class ClickHouseClient:
    """Управление подключением и операциями с ClickHouse."""

    def __init__(self) -> None:
        self.host = settings.CLICKHOUSE_HOST
        self.port = settings.CLICKHOUSE_PORT
        self.user = settings.CLICKHOUSE_USER
        self.password = settings.CLICKHOUSE_PASSWORD
        self.database = settings.CLICKHOUSE_DATABASE
        self._client: Optional[clickhouse_connect.driver.Client] = None
        self._agent_client: Optional[clickhouse_connect.driver.Client] = None

    def _get_client(self) -> clickhouse_connect.driver.Client:
        if self._client is None:
            self._client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                username=self.user,
                password=self.password,
                database=self.database,
                compress=True,
            )
            logger.info("Connected to ClickHouse")
        return self._client

    def _get_agent_client(self) -> clickhouse_connect.driver.Client:
        if self._agent_client is None:
            self._agent_client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                username=settings.AGENT_READONLY_USER,
                password=settings.AGENT_READONLY_PASSWORD,
                database=self.database,
            )
            logger.info("Agent ClickHouse client created")
        return self._agent_client

    def table_exists(self, table_name: str) -> bool:
        try:
            res = self._get_client().query_df(
                f"SELECT name FROM system.tables WHERE database='{self.database}' AND name='{table_name}'"
            )
            return not res.empty
        except Exception:
            return False

    def ensure_tables_for_ticker(self, ticker: str, interval: str) -> None:
        self._create_raw_candles_table(ticker, interval)
        self._create_features_table(ticker, interval)
        self._create_predictions_table(ticker, interval)
        self._create_aggregate_tables_and_mv(ticker, interval)
        logger.info(f"All tables ensured for {ticker}_{interval}")

    def _create_raw_candles_table(self, ticker: str, interval: str) -> None:
        table = get_raw_candles_table(ticker, interval)
        self._get_client().command(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                begin DateTime,
                open Float32,
                high Float32,
                low Float32,
                close Float32,
                volume UInt64,
                value Float32,
                insert_time DateTime DEFAULT now()
            ) ENGINE = ReplacingMergeTree(insert_time)
            ORDER BY (begin)
        """)

    def _create_features_table(self, ticker: str, interval: str) -> None:
        table = get_features_table(ticker, interval)
        self._get_client().command(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                begin DateTime,
                close Float32,
                high Float32,
                low Float32,
                value Float32,
                sma_30 Float32,
                cma_30 Float32,
                ema_30 Float32,
                macd_fast Float32,
                macd_slow Float32,
                macd_signal Float32,
                macd Float32,
                macd_hist Float32
            ) ENGINE = MergeTree()
            ORDER BY (begin)
        """)

    def _create_predictions_table(self, ticker: str, interval: str) -> None:
        table = get_predictions_table(ticker, interval)
        self._get_client().command(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                timestamp DateTime,
                value Float32,
                ticker String,
                interval String,
                is_history UInt8 DEFAULT 0
            ) ENGINE = MergeTree()
            ORDER BY (timestamp)
        """)

    def _create_aggregate_tables_and_mv(self, ticker: str, interval: str) -> None:
        raw_table = get_raw_candles_table(ticker, interval)

        hourly_table = get_hourly_agg_table(ticker, interval)
        self._get_client().command(f"""
            CREATE TABLE IF NOT EXISTS {hourly_table} (
                hour DateTime,
                open_first AggregateFunction(min, Float32),
                open_last AggregateFunction(max, Float32),
                high_max AggregateFunction(max, Float32),
                low_min AggregateFunction(min, Float32),
                close_avg AggregateFunction(avg, Float32),
                volume_sum AggregateFunction(sum, UInt64),
                trade_count AggregateFunction(count)
            ) ENGINE = AggregatingMergeTree()
            PARTITION BY toYYYYMM(hour)
            ORDER BY (hour)
        """)
        self._get_client().command(f"""
            CREATE MATERIALIZED VIEW IF NOT EXISTS {hourly_table}_mv
            TO {hourly_table}
            AS SELECT
                toStartOfHour(begin) AS hour,
                minState(open) AS open_first,
                maxState(open) AS open_last,
                maxState(high) AS high_max,
                minState(low) AS low_min,
                avgState(close) AS close_avg,
                sumState(volume) AS volume_sum,
                countState() AS trade_count
            FROM {raw_table}
            GROUP BY hour
        """)

        daily_table = get_daily_agg_table(ticker, interval)
        self._get_client().command(f"""
            CREATE TABLE IF NOT EXISTS {daily_table} (
                day Date,
                open_first AggregateFunction(min, Float32),
                open_last AggregateFunction(max, Float32),
                high_max AggregateFunction(max, Float32),
                low_min AggregateFunction(min, Float32),
                close_avg AggregateFunction(avg, Float32),
                volume_sum AggregateFunction(sum, UInt64),
                trade_count AggregateFunction(count)
            ) ENGINE = AggregatingMergeTree()
            PARTITION BY toYYYYMM(day)
            ORDER BY (day)
        """)
        self._get_client().command(f"""
            CREATE MATERIALIZED VIEW IF NOT EXISTS {daily_table}_mv
            TO {daily_table}
            AS SELECT
                toDate(begin) AS day,
                minState(open) AS open_first,
                maxState(open) AS open_last,
                maxState(high) AS high_max,
                minState(low) AS low_min,
                avgState(close) AS close_avg,
                sumState(volume) AS volume_sum,
                countState() AS trade_count
            FROM {raw_table}
            GROUP BY day
        """)

    def insert_raw_candles(self, ticker: str, interval: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        table = get_raw_candles_table(ticker, interval)
        df["begin"] = pd.to_datetime(df["begin"])
        min_begin = df["begin"].min()
        max_begin = df["begin"].max()
        self._get_client().command(
            f"DELETE FROM {table} WHERE begin >= '{min_begin}' AND begin <= '{max_begin}'"
        )
        self._get_client().insert_df(table, df)
        logger.info(f"Inserted {len(df)} candles into {table}")

    def insert_features(self, ticker: str, interval: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        table = get_features_table(ticker, interval)
        df["begin"] = pd.to_datetime(df["begin"])
        total = len(df)
        for start in range(0, total, BATCH_SIZE):
            chunk = df.iloc[start:start + BATCH_SIZE]
            begins = chunk["begin"].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
            self._get_client().command(
                f"DELETE FROM {table} WHERE begin IN ({','.join(repr(b) for b in begins)})"
            )
            self._get_client().insert_df(table, chunk)
            logger.debug(f"Inserted features chunk {start // BATCH_SIZE + 1}")
        logger.info(f"Inserted features for {ticker}_{interval}: {total} rows")

    def insert_predictions_batch(self, ticker: str, interval: str,
                                 records: List[Dict[str, Any]]) -> None:
        """Пакетная вставка предсказаний с точным удалением по ticker и interval."""
        if not records:
            return
        table = get_predictions_table(ticker, interval)
        for i in range(0, len(records), BATCH_SIZE):
            chunk = records[i:i + BATCH_SIZE]
            timestamps = [r["timestamp"].strftime('%Y-%m-%d %H:%M:%S') for r in chunk]
            self._get_client().command(
                f"DELETE FROM {table} "
                f"WHERE timestamp IN ({','.join(repr(t) for t in timestamps)}) "
                f"AND ticker = '{ticker}' AND interval = '{interval}'"
            )
            rows = [[r["timestamp"], r["value"], ticker, interval, r.get("is_history", 0)] for r in chunk]
            self._get_client().insert(
                table,
                rows,
                column_names=["timestamp", "value", "ticker", "interval", "is_history"]
            )
            logger.debug(f"Inserted predictions chunk {i // BATCH_SIZE + 1}")

    def get_last_candle_date(self, ticker: str, interval: str) -> Optional[pd.Timestamp]:
        table = get_raw_candles_table(ticker, interval)
        res = self._get_client().query_df(f"SELECT max(begin) AS last FROM {table}")
        if res.empty:
            return None
        val = res["last"][0]
        if pd.isna(val):
            return None
        ts = pd.Timestamp(val)
        if ts < pd.Timestamp(MIN_VALID_DATE):
            return None
        return ts

    def get_candles(self, ticker: str, interval: str,
                    from_date: Optional[datetime] = None,
                    to_date: Optional[datetime] = None,
                    limit: int = 1000) -> pd.DataFrame:
        table = get_raw_candles_table(ticker, interval)
        conditions = ["1=1"]
        if from_date:
            conditions.append(f"begin >= '{from_date}'")
        if to_date:
            conditions.append(f"begin <= '{to_date}'")
        where = " AND ".join(conditions)
        query = f"""
            SELECT begin, open, high, low, close, volume, value
            FROM {table}
            WHERE {where}
            ORDER BY begin DESC
            LIMIT {limit}
        """
        return self._get_client().query_df(query)

    def get_predictions(self, ticker: str, interval: str, limit: int = 100) -> pd.DataFrame:
        table = get_predictions_table(ticker, interval)
        query = f"""
            SELECT timestamp, value, ticker, interval
            FROM {table}
            WHERE ticker='{ticker}' AND interval='{interval}'
            ORDER BY timestamp DESC LIMIT {limit}
        """
        return self._get_client().query_df(query)

    def get_last_features_context(self, ticker: str, interval: str, size: int = 30) -> pd.DataFrame:
        table = get_features_table(ticker, interval)
        if not self.table_exists(table):
            return pd.DataFrame(columns=[
                "begin", "close", "high", "low", "value",
                "sma_30", "cma_30", "ema_30",
                "macd_fast", "macd_slow", "macd_signal",
                "macd", "macd_hist"
            ])
        query = f"""
            SELECT *
            FROM {table}
            ORDER BY begin DESC
            LIMIT {size}
        """
        df = self._get_client().query_df(query)
        if df.empty:
            return pd.DataFrame(columns=[
                "begin", "close", "high", "low", "value",
                "sma_30", "cma_30", "ema_30",
                "macd_fast", "macd_slow", "macd_signal",
                "macd", "macd_hist"
            ])
        return df.sort_values("begin")

    def get_features_for_range(self, ticker: str, interval: str,
                               start: datetime, end: datetime) -> pd.DataFrame:
        table = get_features_table(ticker, interval)
        query = f"""
            SELECT * FROM {table}
            WHERE begin >= '{start}' AND begin <= '{end}'
            ORDER BY begin
        """
        return self._get_client().query_df(query)

    def execute_readonly_query(self, query: str) -> Dict[str, Any]:
        logger.info(f"Executing read-only SQL: {query}")
        df = self._get_agent_client().query_df(query)
        records = df.to_dict(orient="records")
        columns = df.columns.tolist()
        rows = [list(rec.values()) for rec in records]
        return {"columns": columns, "rows": rows, "row_count": len(rows)}


db_client = ClickHouseClient()