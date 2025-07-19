from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from pyannote.audio import Pipeline
import tempfile
import hashlib
from dotenv import load_dotenv
import os

load_dotenv()
# Ваш захешированный API ключ (используй безопасный способ хранения в проде!)
API_KEY = os.getenv("API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@app.post("/diarize")
async def diarize_audio(file: UploadFile = File(...), x_api_key: str = Header(...)):
    if hash_key(x_api_key) != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()

        diarization = pipeline(tmp.name)

        merged_result = []
        previous = None

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = turn.start
            end = turn.end

            if previous is None:
                previous = {"start": start, "end": end, "speaker": speaker}
                continue

            if speaker == previous["speaker"] and abs(start - previous["end"]) < 0.5:
                previous["end"] = end
            else:
                merged_result.append(previous)
                previous = {"start": start, "end": end, "speaker": speaker}

        if previous:
            merged_result.append(previous)

    return {"diarization": merged_result}
