from catboost import CatBoostRegressor
import clickhouse_connect
import asyncio
import requests
import apimoex
import datetime
import pandas as pd
import numpy as np
import multiprocessing
import time
from fastapi import FastAPI
from pydantic import BaseModel



class bot_api:
    def __init__(self, CLICKHOUSE_CLOUD_HOSTNAME: str = '10.0.2.15', CLICKHOUSE_CLOUD_USER: str = 'utopialvo', CLICKHOUSE_CLOUD_PASSWORD: str = 'utopialvo'):
        self.CLICKHOUSE_CLOUD_HOSTNAME = CLICKHOUSE_CLOUD_HOSTNAME
        self.CLICKHOUSE_CLOUD_USER = CLICKHOUSE_CLOUD_USER
        self.CLICKHOUSE_CLOUD_PASSWORD = CLICKHOUSE_CLOUD_PASSWORD
        self.client = clickhouse_connect.get_client(host=self.CLICKHOUSE_CLOUD_HOSTNAME, port=8123, username=self.CLICKHOUSE_CLOUD_USER, password=self.CLICKHOUSE_CLOUD_PASSWORD, query_limit = 0)
        self.reg = CatBoostRegressor()
        self.reg.load_model("reg.model")
        
    def download_all_candles(self, name, interval: int = 1):
        fulldata = []
        for i in list(zip([f'20{str(11+i)}-01-01 10:00:00' for i in range(14)],[f'20{str(12+i)}-01-01 10:00:00' for i in range(14)])):
            with requests.session() as session:
                fulldata.extend(apimoex.get_market_candles(session, name, interval=interval, start=i[0], end=i[1]))
        return fulldata
    
    def download_data(self, name: str, interval: int = 1):
        tablename = name + '_interval_' + str(interval)
        start = self.client.query(f'SELECT max(begin) FROM {tablename}').result_rows[0][0]
        start = (datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S") + datetime.timedelta(minutes=interval)).strftime("%Y-%m-%d %H:%M:%S")
        end = (datetime.datetime.now() + datetime.timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
        with requests.session() as session:
                fulldata = apimoex.get_market_candles(session, name, interval=interval, start=start, end=end)
        return fulldata
    
    def create_table(self, name: str, interval: int = 1):
        tablename = name + '_interval_' + str(interval)
        self.client.command(f'CREATE TABLE IF NOT EXISTS {tablename} (begin DateTime("Europe/Moscow"), open Float32, close Float32, high Float32, low Float32, value Float64, volume UInt64) ENGINE MergeTree() ORDER BY (begin)')
        self.client.command(f'CREATE TABLE IF NOT EXISTS {tablename}_pred (id UInt64, begin DateTime("Europe/Moscow"), close Float32) ENGINE MergeTree() ORDER BY (begin)')
        print(f"table {tablename} created or exists already!\n")
        return None
    
    def drop_table(self, tablename: str):
        self.client.query(f'DROP TABLE IF EXISTS {tablename}')
        return None
    
    def insert_data_to_table(self, name: str, chunk: list, interval: int = 1, flag: bool = True):
        tablename = name + '_interval_' + str(interval)
        if flag == True:
            self.client.insert(table = tablename, data = chunk, column_names=['begin', 'open', 'close', 'high', 'low', 'value', 'volume'])
        else:
            tablename = tablename + '_pred'
            self.client.insert(table = tablename, data = chunk, column_names=['id','begin', 'close'])
        print(f"written {len(chunk)} rows to table {tablename}\n")
        return None
    
    def get_candles_dataframe(self, name, interval:int = 1):
        name = name + '_interval_' + str(interval)
        df = self.client.query_df(f'SELECT * FROM default.{name}')
        df = df.sort_values('begin', ascending=True).reset_index(drop=True)
        return df
    
    def get_candles_dataframe_mat(self, name, interval:int = 1):
        name = name + '_interval_' + str(interval) + '_view'
        df = self.client.query_df(f'SELECT * FROM {name}')
        df = df.sort_values('begin', ascending=True).reset_index(drop=True)
        return df
    
    def prep_chunk_data(self, data: list):
        chunk = []
        for i in data:
            chunk.append(list(i.values()))
        return chunk
    
    def create_mater_view(self, name:str, interval: int = 1):
        name = name + '_interval_' + str(interval)
        self.client.command(f'CREATE TABLE IF NOT EXISTS {name}_mat (begin DateTime("Europe/Moscow"), open Float32, close Float32, high Float32, low Float32, value Float64, volume UInt64, sma_30 Float32, cma_30 Float32, ema_30 Float32, macd_fast Float32, macd_slow Float32, macd_signal Float32, macd Float32, macd_hist Float32, id UInt64) ENGINE AggregatingMergeTree() PARTITION BY toYYYYMM(begin) ORDER BY (begin)')   
        self.client.query(f"""
        CREATE MATERIALIZED VIEW {name}_view TO {name}_mat AS
        SELECT
          *,
          avg(close) OVER (ORDER BY begin ROWS 29 PRECEDING) AS sma_30,
          avg(close) OVER (ORDER BY begin ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS cma_30,
          avg(close) OVER (ORDER BY begin ROWS 29 PRECEDING) * 0.967 + close * 0.033 AS ema_30,
          avg(close) OVER (ORDER BY begin ROWS 12 PRECEDING) AS macd_fast,
          avg(close) OVER (ORDER BY begin ROWS 26 PRECEDING) AS macd_slow,
          avg(close) OVER (ORDER BY begin ROWS 9 PRECEDING) AS macd_signal,
          macd_fast - macd_slow AS macd,
          macd_fast - macd_signal AS macd_hist,
          row_number() OVER (ORDER BY begin) AS id
        FROM 
          {name}
        ORDER BY
          begin
        """)
        print(f"table {name} created or exists already!\n")
        return None
    
    def predict_candles(self, name, interval: int = 1, ch: bool = False):
        temp = self.client.query_df(f"""
        SELECT *
        FROM {name}_interval_{str(interval)}_view
        WHERE begin = (SELECT max(begin) FROM {name}_interval_{str(interval)}_view)
        """).drop(['volume','open','id'],axis=1)
        dt = self.client.query(f'SELECT max(id) FROM {name}_interval_{str(interval)}_view WHERE begin = (SELECT max(begin) FROM {name}_interval_{str(interval)}_view)').result_rows[0][0]
        begin = temp.begin.values[0]
        if ch == True: 
            tablename = name + '_interval_' + str(interval) + '_pred'
            check = self.client.query(f'SELECT max(id) FROM {tablename}').result_rows[0][0]
            if check > dt:
                return None
        temp = temp.set_index('begin')
        pred = self.reg.predict(temp)[0]
        return {'id': int(dt+1),"begin": begin, 'close':pred}




app = FastAPI()
all_processes = {}


class TaskOptionBody(BaseModel): 
    name: str
    interval: int

@app.post('/task/run')
def task_run(TaskOption: TaskOptionBody):
    process = multiprocessing.Process(target=task, args=(TaskOption.name,TaskOption.interval))
    process.start()
    global all_processes
    all_processes[f'{TaskOption.name}_{TaskOption.interval}'] = process
    return 0

@app.get('/task/abort')
def task_abort(TaskOption: TaskOptionBody):
    global all_processes
    all_processes.get(f'{TaskOption.name}_{TaskOption.interval}').terminate()
    return 0


def task(name :str, interval: int = 1):
    job = bot_api()
    while True:
        pred = job.predict_candles(name=name, interval = interval, ch = True)
        if type(pred) != type(None):
            pred = job.prep_chunk_data([pred])
            job.insert_data_to_table(name = name, interval = interval, chunk = pred, flag = False)
        else:
            print(f'{name} pred not inserted ')
        d = job.download_data(name = name, interval = interval)
        if len(d) > 0:
            d = job.prep_chunk_data(d)
            job.insert_data_to_table(name = name, interval = interval, chunk = d)
        else:
            print(f'{name} new data not inserted')
        if interval == 1:
            time.sleep(60)
        elif interval == 10:
            time.sleep(600)


if __name__ == "__main__":
	uvicorn.run(app, host='0.0.0.0', port=80)
    
