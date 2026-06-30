# Файл: dockerfile
# Сборка образа приложения MOEX_BOT
FROM python:3.12-slim-bookworm

WORKDIR /app

# Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Создаём директорию для моделей
RUN mkdir -p /app/models

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]