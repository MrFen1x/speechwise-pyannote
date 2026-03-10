FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --fix-missing \
    ffmpeg \
    libsndfile1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Копирование проекта
WORKDIR /app
COPY requirements.txt .

# Установка зависимостей (Torch + requirements.txt) одной командой для предотвращения конфликтов
RUN pip install --upgrade pip && \
    pip install --prefer-binary --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    python-dotenv numpy==1.26.4 python-multipart \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

# Открываем порт
EXPOSE 5671

# Запуск приложения
CMD ["uvicorn", "diarization_service:app", "--host", "0.0.0.0", "--port", "5671"]
