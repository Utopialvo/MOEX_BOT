FROM python:3.12-slim-bookworm 

COPY server.py /app/
COPY requirements.txt /app/
COPY reg.model /app/

WORKDIR /app/

RUN pip install -r requirements.txt

EXPOSE 80

CMD ["uvicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "80"]
