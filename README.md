# MOEX_BOT

Сервис для прогнозирования котировок Московской биржи.
Может в реальном времени собирать свечи, считать признаки и строить прогнозы
с помощью моделей CatBoost. Всё складывается в ClickHouse, а наружу торчит REST API.

## Что умеет

- Загружать исторические свечи с MOEX (1m и 10m) с 2000 года.
- Автоматически докачивать новые данные и обновлять прогнозы.
- На лету рассчитывать признаки: SMA, EMA, MACD и др.
- Строить предсказания цен на основе загруженных моделей `.cbm`.
- Считать агрегаты (часовые, дневные) и торговые сигналы.
- Отдавать данные через API: свечи, прогнозы, агрегаты, сигналы, корреляции.
- Позволять агентам выполнять SQL-запросы только на чтение (от имени ограниченного пользователя ClickHouse).
- Поддерживать массовые запросы (batch) для нескольких тикеров.
- Автоматически перезапускать упавшие задачи, пока не остановлены явно.

Внутри всё упаковано в Docker Compose: поднимается ClickHouse и сам сервис на FastAPI.

## Быстрый старт

1. Клонируй репозиторий.
2. Создай `.env` на основе `.env.example` (можно ничего не менять для пробы).
3. Положи модели в папку `./models`. Хотя бы одну — дефолтную `reg.model`.
4. Запусти:
   ```bash
   docker compose up -d
   ```
5. Сервис будет доступен на `http://localhost:80`.

## Конфигурация

Все настройки через переменные окружения (`.env`):

| Переменная                 | Назначение                                         | По умолчанию          |
| -------------------------- | -------------------------------------------------- | --------------------- |
| `CLICKHOUSE_HOST`          | Хост ClickHouse (внутри сети)                      | `localhost`           |
| `CLICKHOUSE_PORT`          | HTTP-порт ClickHouse                               | `8123`                |
| `CLICKHOUSE_USER`          | Пользователь ClickHouse (admin)                    | `default`             |
| `CLICKHOUSE_PASSWORD`      | Пароль пользователя                                | пусто                 |
| `CLICKHOUSE_DATABASE`      | База данных                                        | `default`             |
| `MODEL_DIR`                | Папка с моделями внутри контейнера                 | `/app/models`         |
| `DEFAULT_MODEL_PATH`       | Путь к дефолтной модели                            | `/app/models/reg.model` |
| `MAX_CONCURRENT_TASKS`     | Максимум одновременно работающих задач             | `10`                  |
| `PAUSE_1M` / `PAUSE_10M`   | Пауза между обновлениями в секундах                | `60` / `600`          |
| `LOG_LEVEL`                | Уровень логирования (`DEBUG`, `INFO`, `WARNING`)   | `INFO`                |
| `AGENT_READONLY_USER`      | Имя пользователя для агентов (read-only)           | `agent_ro`            |
| `AGENT_READONLY_PASSWORD`  | Пароль read-only пользователя                      | `secure_pass`         |

## Основные эндпоинты API

Все пути начинаются с `/api/v1`. Документация доступна по `/docs` (Swagger).

### Управление задачами

- `POST /api/v1/tasks` — запустить задачу для тикера.
  ```json
  { "ticker": "GAZP", "interval": "1m", "model_name": "gazp_model" }
  ```
  `model_name` — необязательное поле, при отсутствии используется дефолтная модель или модель `{ticker}.cbm`.

- `GET /api/v1/tasks` — список всех задач.
- `GET /api/v1/tasks/{task_id}` — статус конкретной задачи.
- `DELETE /api/v1/tasks/{task_id}` — остановить задачу.
- `POST /api/v1/tasks/{task_id}/restart` — перезапустить.
- `POST /api/v1/tasks/{task_id}/recalc-predictions` — запустить фоновый полный пересчёт истории (воркер приостанавливается на время пересчёта).

### Данные

- `GET /api/v1/data/candles/{ticker}?interval=1m&limit=100` — свечи.
- `GET /api/v1/data/predictions/{ticker}?interval=1m` — последнее предсказание.
- `GET /api/v1/data/predictions/{ticker}/history?interval=1m&limit=50` — история прогнозов.
- `GET /api/v1/data/aggregates/{ticker}?interval=1m&granularity=hourly` — агрегаты.
- `GET /api/v1/data/signals/{ticker}?interval=1m` — торговые сигналы (SMA-12/26).
- `GET /api/v1/data/correlations?tickers=GAZP,SBER&interval=10m&window=100` — корреляции.
- `POST /api/v1/data/predictions/batch` — батчевый запрос прогнозов.
- `POST /api/v1/data/candles/batch` — батчевый запрос свечей.

### Модели

- `GET /api/v1/models` — список доступных моделей.
- `POST /api/v1/models/upload/{ticker}` — загрузить `.cbm` файл (сохранится как `{ticker}.cbm`).
- `DELETE /api/v1/models/{ticker}` — удалить модель.

### SQL для агентов

- `POST /api/v1/sql/query` — выполнить SELECT-запрос под пользователем `agent_ro`.
  ```json
  { "query": "SELECT count() FROM raw_candles_GAZP_1m" }
  ```
  Разрешены только запросы, начинающиеся с `SELECT`.

### Система

- `GET /api/v1/system/health` — проверка здоровья.
- `GET /api/v1/system/metrics` — количество задач, свечей и последняя ошибка.
- `GET /api/v1/system/info` — метаданные для агентов (активные тикеры, модели, паузы).

## Как добавить тикер и начать получать прогнозы

1. Положи в папку `./models` модель с именем `<тикер>.cbm`, например `GAZP.cbm`. Если её нет, будет использована дефолтная (`reg.model`).
2. Отправь запрос:
   ```bash
   curl -X POST http://localhost:80/api/v1/tasks \
        -H 'Content-Type: application/json' \
        -d '{"ticker":"GAZP","interval":"1m"}'
   ```
3. Сервис создаст таблицы в ClickHouse, загрузит всю доступную историю свечей (с 2000 года), посчитает признаки и первые прогнозы. Дальше будет обновляться каждые 60 секунд (для 1m) и докладывать свежие данные.
4. Посмотри прогноз:
   ```bash
   curl http://localhost:80/api/v1/data/predictions/GAZP?interval=1m
   ```

## Как работают воркеры и автоматическое восстановление

Каждая задача — это отдельный фоновый поток (не блокирует event loop). Если воркер падает 10 раз подряд, он останавливается. Но если задача не была остановлена явно (через `DELETE /tasks/{task_id}`), менеджер попытается перезапустить её с нарастающей задержкой (10с, 20с, … до 10 минут). Это позволяет переживать временные проблемы с MOEX или ClickHouse.

Остановленная вручную задача перезапускаться не будет.


## Структура проекта

```
MOEX_BOT/
├── app/                       # Код приложения FastAPI
│   ├── api/v1/                # Роутеры API
│   ├── core/                  # Константы, исключения, логирование
│   ├── models/                # Pydantic-схемы
│   ├── services/              # Клиент ClickHouse, MOEX, модели, фичи
│   └── workers/               # Воркер прогнозирования и менеджер задач
├── fs/volumes/clickhouse/     # Конфиги ClickHouse (монтируются)
├── models/                    # Файлы моделей (.cbm)
├── docker-compose.yml
├── dockerfile
├── .env.example
└── requirements.txt           # Python-зависимости (не забудь создать)
```

## Пару слов про модели

Модели ожидаются в формате CatBoost (`.cbm`). Имя файла должно быть либо `reg.model` для дефолтной, либо `<ticker>.cbm` для конкретного инструмента. При старте задачи можно явно указать `model_name` — тогда будет использован файл `{model_name}.cbm`. Это удобно, если есть разные версии моделей для одного тикера.

Признаковый состав фиксирован (поля `FEATURE_COLUMNS` в `services/model.py`): `close`, `high`, `low`, `value`, `sma_30`, `cma_30`, `ema_30` и пять компонент MACD.