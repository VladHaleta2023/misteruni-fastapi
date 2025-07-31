from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from faster_whisper import WhisperModel
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
import tempfile
import subprocess
import wave
import spacy
import os
import re
from fastapi.responses import FileResponse
from gtts import gTTS
import uuid
import boto3
import urllib.parse
from io import BytesIO
import copy

load_dotenv()
port = int(os.getenv("PORT", 4200))

app = FastAPI()

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

BUCKET_NAME = os.getenv("AWS_BUCKET")
REGION = os.getenv("AWS_REGION")

whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    '.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.wma', '.opus',
    '.amr', '.aiff', '.alac', '.pcm', '.webm', '.mp4', '.3gp', '.caf'
}

MAX_AUDIO_DURATION = 900

spacy_models = {
    'pl': 'pl_core_news_sm',
    'ru': 'ru_core_news_sm',
    'en': 'en_core_web_sm',
}

MAX_ATTEMPTS = 10

class PromptRequest(BaseModel):
    prompt: str

class ImagePromptRequest(BaseModel):
    image_url: str

class SplitIntoSentencesRequest(BaseModel):
    text: str
    language: Optional[str]

class SplitIntoSentencesResponse(BaseModel):
    sentences: list

class TranscriptionPartResponse(BaseModel):
    part_id: int
    transcription: str
    language: str
    language_probability: Optional[float]
    subject: Optional[str]

class TTSRequest(BaseModel):
    id: int
    part_id: int
    text: str
    language: str = "ru"

class SubtopicsGenerator(BaseModel):
    changed: str
    subject: str
    section: str
    topic: str
    subtopics: List[str]
    attempt: int
    prompt: str

@app.get("/")
def root():
    return {"message": f"Serwer działa na porcie {port}"}


@app.post("/admin/full-plan-generate")
def full_plan_generate(data: PromptRequest):
    try:
        from plan_generator import full_plan_generate
        return full_plan_generate(data.prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


def is_allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXTENSIONS


def convert_to_wav(input_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as input_tmp:
        input_tmp.write(input_bytes)
        input_tmp_path = input_tmp.name

    output_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_tmp_path = output_tmp.name
    output_tmp.close()

    cmd = [
        "ffmpeg", "-y", "-i", input_tmp_path,
        "-ar", "16000", "-ac", "1",
        "-f", "wav", output_tmp_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.remove(input_tmp_path)

    if result.returncode != 0:
        os.remove(output_tmp_path)
        raise HTTPException(status_code=400, detail="Błąd konwersji pliku audio (ffmpeg).")

    return output_tmp_path


@app.post("/admin/audio-transcribe-part", response_model=TranscriptionPartResponse)
async def transcribe_audio_part(
        file: UploadFile = File(...),
        part_id: int = Form(...),
        subject: Optional[str] = Form(None),
        language: Optional[str] = Form(None)
):
    language = language or 'ru'
    subject = subject or 'Brak przedmiotu'
    filename = file.filename.lower()

    if not is_allowed_file(filename):
        raise HTTPException(status_code=400, detail="Nieobsługiwany format pliku audio.")

    audio_bytes = await file.read()
    if len(audio_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Plik audio jest zbyt duży. Maksymalnie 100 MB.")

    try:
        wav_path = convert_to_wav(audio_bytes)

        with wave.open(wav_path, "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)

        if duration > MAX_AUDIO_DURATION:
            os.remove(wav_path)
            raise HTTPException(
                status_code=400,
                detail=f"Plik audio jest za długi: {duration:.2f} sekund. Maksymalnie 30 minut."
            )

        segments, info = whisper_model.transcribe(
            wav_path,
            beam_size=5,
            language=language,
            temperature=0.2
        )
        transcription = " ".join(segment.text.strip() for segment in segments).strip()
        os.remove(wav_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd serwera: {str(e)}")

    return TranscriptionPartResponse(
        part_id=int(part_id),
        transcription=str(transcription),
        language=str(info.language),
        language_probability=float(round(info.language_probability, 2)) if info.language_probability is not None else None,
        subject=str(subject) if subject else None
    )


def split_text_into_sentences(text: str, language: str = 'ru'):
    model_name = spacy_models.get(language, 'ru_core_news_sm')
    try:
        nlp = spacy.load(model_name)
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences if sentences else [text.strip()]
    except Exception:
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


@app.post("/admin/split-into-sentences", response_model=SplitIntoSentencesResponse)
def split_into_sentences(data: SplitIntoSentencesRequest):
    try:
        return SplitIntoSentencesResponse(sentences=split_text_into_sentences(data.text, data.language))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd serwera: {str(e)}")


@app.post("/admin/tts")
def generate_tts(data: TTSRequest):
    try:
        filename = f"tts_{data.id}_{data.part_id}_{uuid.uuid4()}.mp3"
        mp3_fp = BytesIO()

        tts = gTTS(text=data.text, lang=data.language)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        s3.upload_fileobj(mp3_fp, BUCKET_NAME, filename, ExtraArgs={"ContentType": "audio/mpeg"})

        public_url = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{urllib.parse.quote(filename)}"
        return {"url": public_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd generacji TTS lub upload do S3: {str(e)}")

@app.post("/admin/subtopics-generate")
def subtopics_generate(data: SubtopicsGenerator):
    try:
        from criterion_generator import (
            request_ai,
            metadata_fill_template,
            extract_format_from_prompt,
            parse_output_by_format,
            detect_changes_by_format,
            clean_output
        )

        old_data = copy.deepcopy(data.dict())

        if data.attempt > MAX_ATTEMPTS or data.changed != 'true':
            return SubtopicsGenerator(**old_data)

        prompt = metadata_fill_template(data.prompt, data.dict())
        response = request_ai(prompt)

        if not response:
            raise HTTPException(status_code=500, detail="AI Response is None")

        response = clean_output(response)
        format_str = extract_format_from_prompt(prompt)

        new_data = parse_output_by_format(format_str, response, data.dict())
        new_data['attempt'] += 1

        if detect_changes_by_format(old_data, new_data, format_str):
            new_data['changed'] = 'true'
        else:
            new_data['changed'] = 'false'

        return SubtopicsGenerator(**new_data)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        timeout_keep_alive=900,
        timeout_graceful_shutdown=900
    )