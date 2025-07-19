FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Установка torch (CPU)
RUN pip install --upgrade pip && \
    pip install --prefer-binary --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir python-dotenv numpy==1.26.4 python-multipart

# Копирование проекта
WORKDIR /app
COPY . /app

# Установка зависимостей проекта
RUN pip install -r requirements.txt

# Открываем порт
EXPOSE 5671

# Запуск приложения
CMD ["uvicorn", "diarization_service:app", "--host", "0.0.0.0", "--port", "5671"]
