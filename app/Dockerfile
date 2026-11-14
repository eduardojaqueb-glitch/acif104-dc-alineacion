FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8000
EXPOSE 8000
# 2 procesos x 4 hilos: la API no guarda estado en memoria, así que escala horizontalmente agregando réplicas
CMD ["gunicorn", "-w", "2", "--threads", "4", "-b", "0.0.0.0:8000", "app:app"]
