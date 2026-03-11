from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from pyannote.audio import Pipeline
import tempfile
import hashlib
from dotenv import load_dotenv
import os
import torch  # <-- ДОБАВЛЕНО: Импортируем PyTorch для работы с GPU

load_dotenv()

# Ваш захешированный API ключ (используй безопасный способ хранения в проде!)
API_KEY = os.getenv("API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

# Загружаем базовую модель с Hugging Face
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

# <-- ДОБАВЛЕНО: Блок для переноса модели на видеокарту
# Проверяем, доступна ли видеокарта (CUDA)
if torch.cuda.is_available():
    # Если доступна, принудительно отправляем модель на GPU
    pipeline.to(torch.device("cuda"))
    print("Успех: Модель загружена на GPU (видеокарту)!")
else:
    # Если видеокарта не найдена, предупреждаем об этом
    print("Внимание: GPU не найдена. Модель будет использовать CPU (процессор).")


def hash_key(key: str) -> str:
    """Хеширует строку для безопасного сравнения (если потребуется в будущем)."""
    return hashlib.sha256(key.encode()).hexdigest()


@app.post("/diarize")
async def diarize_audio(file: UploadFile = File(...), x_api_key: str = Header(...)):
    """
    Принимает аудиофайл, проверяет API-ключ и выполняет разделение по голосам (диаризацию).
    """
    # Простая проверка ключа доступа
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Создаем временный файл для сохранения загруженного аудио
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()

        # Запускаем процесс диаризации на нашем временном файле
        # Если ранее мы перенесли pipeline на GPU, этот процесс пойдет через видеокарту
        pipeline.to(torch.device("cuda"))
        diarization = pipeline(tmp.name)

        # 2. СРАЗУ ВОЗВРАЩАЕМ НА CPU
        pipeline.to(torch.device("cpu"))

        # 3. ОЧИЩАЕМ ПАМЯТЬ ДЛЯ LOCALAI
        torch.cuda.empty_cache()
        import gc; gc.collect()

        merged_result = []
        previous = None

        # Проходим по всем найденным репликам спикеров
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = turn.start
            end = turn.end

            # Сохраняем первую реплику
            if previous is None:
                previous = {"start": start, "end": end, "speaker": speaker}
                continue

            # Если говорит тот же спикер и пауза меньше 0.5 секунд, объединяем реплики
            if speaker == previous["speaker"] and abs(start - previous["end"]) < 0.5:
                previous["end"] = end
            else:
                # Если спикер сменился или пауза большая, сохраняем предыдущую фразу и начинаем новую
                merged_result.append(previous)
                previous = {"start": start, "end": end, "speaker": speaker}

        # Не забываем добавить последнюю реплику
        if previous:
            merged_result.append(previous)
    
    return {"diarization": merged_result}
