FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Установка Python 3.10 и системных зависимостей
RUN apt-get update && apt-get install -y --fix-missing \
    python3.10 \
    python3-pip \
    python3.10-dev \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Создание символической ссылки на python3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Копирование проекта
WORKDIR /app
COPY requirements.txt .

# Установка зависимостей (Torch + requirements.txt) одной командой для предотвращения конфликтов
# Используем CUDA версию PyTorch
RUN pip install --upgrade pip && \
    pip install --prefer-binary --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    python-dotenv numpy==1.26.4 python-multipart \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118

COPY . .

# Открываем порт
EXPOSE 5671

# Запуск приложения
CMD ["uvicorn", "diarization_service:app", "--host", "0.0.0.0", "--port", "5671"]
