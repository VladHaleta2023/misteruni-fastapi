from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body, Request
#from faster_whisper import WhisperModel
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from dotenv import load_dotenv
#import tempfile
#import subprocess
#import wave
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
import json
import logging
import sys
import asyncio
import random
from g4f.client import Client

client = Client()

top_models = [
    "deepseek-v3",   # очень сильная reasoning-модель
    "gpt-4",
    "gpt-4.1",       # обновлённый GPT-4 с улучшенным reasoning
    "gpt-4.5",       # топовая версия GPT-4 (лучше всего для сложных задач)
    "deepseek-r1",   # ещё более мощная reasoning LLM
    "deepseek-r1-turbo",  # оптимизированная версия для reasoning
    "deepseek-v3-0324",  # оптимизированная версия
    "deepseek-v3-0324-turbo",
    "deepseek-r1-0528",
    "deepseek-r1-0528-turbo",
    "gpt-4o-mini",   # компактнее, но всё ещё хорошо справляется
    "gpt-4.1-mini",  # более быстрый и дешёвый вариант
    "gpt-4.1-nano",  # лёгкая версия, но справляется с текстовыми задачами
    "grok-3",        # от xAI, сильный reasoning
    "grok-3-r1"      # reasoning-ориентированный Grok
]

logger = logging.getLogger("app_logger")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

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

#whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

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

nlp_models: Dict[str, Optional[spacy.language.Language]] = {}

@app.on_event("startup")
async def load_spacy_models():
    for lang, model_name in spacy_models.items():
        try:
            nlp_models[lang] = spacy.load(model_name)
            logger.info(f"Model SpaCy dla {lang} jest pobrana: {model_name}")
        except Exception as e:
            logger.warning(f"Nie udało się podrać model {model_name} dla {lang}: {e}")
            nlp_models[lang] = None

MAX_ATTEMPTS = 2

def fill_placeholders(prompt: str, data: Dict[str, Any]) -> str:
    def replacer(match: re.Match) -> str:
        key = match.group(1)
        value = data.get(key, "")
        if isinstance(value, list):
            if not value:
                return f"{key}Start:\n{key}End:"
            lines = []
            for item in value:
                if isinstance(item, list):
                    lines.append(";".join(str(subitem) for subitem in item))
                else:
                    lines.append(str(item))
            return f"{key}Start:\n" + "\n".join(lines) + f"\n{key}End:"
        return str(value)
    pattern = r"\{\$(\w+)\$\}"
    return re.sub(pattern, replacer, prompt)

async def request_ai(prompt: str, data: Dict[str, Any], request: Request) -> Optional[str]:
    prompt_filled = fill_placeholders(prompt, data)
    abort_event = asyncio.Event()

    logger.info("Prompt:\n" + prompt_filled)

    async def monitor_disconnect():
        while not abort_event.is_set():
            if await request.is_disconnected():
                logger.info("Client disconnected, aborting AI call")
                abort_event.set()
            await asyncio.sleep(0.1)

    async def call_model(model: str) -> Optional[str]:
        logger.info(f"Model {model} rozpoczęła przetwarzanie zapytania.")

        if abort_event.is_set():
            raise HTTPException(status_code=499, detail="Client disconnected")

        try:
            resp = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt_filled}],
                        temperature=0,
                        web_search=False,
                        stream=False,
                    )
                ),
                timeout=60
            )

            if abort_event.is_set():
                raise HTTPException(status_code=499, detail="Client disconnected")

            content = resp.choices[0].message.content.strip()

            logger.info(f"✅ Model {model} zwróciła poprawny wynik.")
            return content

        except asyncio.TimeoutError:
            logger.error(f"⏳ Model {model} przekroczyła limit czasu 60s.")
            await asyncio.sleep(random.uniform(0.5, 1.5))
            return None
        except Exception as e:
            logger.error(f"❌ Model {model} nie powiodła się z błędem: {e}")
            await asyncio.sleep(random.uniform(0.5, 1.5))
            return None

    monitor_task = asyncio.create_task(monitor_disconnect())

    try:
        for model in top_models:
            if abort_event.is_set():
                raise HTTPException(status_code=499, detail="Client disconnected")

            result = await call_model(model)
            if result is not None:
                return result

        logger.info("Wszystkie modele nie powiodły się lub nie zwróciły wyniku.")
        await asyncio.sleep(random.uniform(0.5, 1.5))
        return None
    finally:
        monitor_task.cancel()

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
    subtopics: List[List]
    attempt: int
    prompt: str
    errors: List[str]

class TaskGenerator(BaseModel):
    changed: str
    subject: str
    section: str
    topic: str
    subtopics: List[List]
    outputSubtopics: List[str]
    difficulty: int
    threshold: int
    text: str
    note: str
    attempt: int
    prompt: str
    errors: List[str]

class InteractiveTaskGenerator(BaseModel):
    changed: str
    subject: str
    section: str
    topic: str
    subtopics: List[List]
    difficulty: int
    text: str
    translate: str
    attempt: int
    prompt: str
    errors: List[str]

class QuestionsTaskGenerator(BaseModel):
    changed: str
    subject: str
    section: str
    topic: str
    difficulty: int
    text: str
    questions: List[str]
    attempt: int
    prompt: str
    errors: List[str]

class SolutionGenerator(BaseModel):
    changed: str
    text: str
    solution: str
    attempt: int
    prompt: str
    errors: List[str]

class OptionsGenerator(BaseModel):
    changed: str
    text: str
    solution: str
    options: List[str]
    explanations: List[str]
    correctOptionIndex: int
    attempt: int
    prompt: str
    errors: List[str]

class ProblemsGenerator(BaseModel):
    changed: str
    text: str
    explanation: str
    solution: str
    options: List[str]
    subtopics: List[str]
    correctOptionIndex: int
    outputSubtopics: List[list]
    difficulty: int
    subject: str
    section: str
    topic: str
    userSolution: str
    userOptionIndex: int
    attempt: int
    prompt: str
    errors: List[str]

class WordsGenerator(BaseModel):
    changed: str
    text: str
    words: List[List]
    outputWords: List[str]
    outputText: str
    attempt: int
    prompt: str
    errors: List[str]

@app.get("/")
async def root():
    return {"message": f"Serwer działa na porcie {port}"}


@app.post("/admin/full-plan-generate")
def full_plan_generate(data: PromptRequest):
    try:
        from plan_generator import full_plan_generate
        return full_plan_generate(data.prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


# def is_allowed_file(filename: str) -> bool:
#     ext = os.path.splitext(filename.lower())[1]
#     return ext in ALLOWED_EXTENSIONS
#
#
# def convert_to_wav(input_bytes: bytes) -> str:
#     with tempfile.NamedTemporaryFile(delete=False) as input_tmp:
#         input_tmp.write(input_bytes)
#         input_tmp_path = input_tmp.name
#
#     output_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#     output_tmp_path = output_tmp.name
#     output_tmp.close()
#
#     cmd = [
#         "ffmpeg", "-y", "-i", input_tmp_path,
#         "-ar", "16000", "-ac", "1",
#         "-f", "wav", output_tmp_path
#     ]
#     result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     os.remove(input_tmp_path)
#
#     if result.returncode != 0:
#         os.remove(output_tmp_path)
#         raise HTTPException(status_code=400, detail="Błąd konwersji pliku audio (ffmpeg).")
#
#     return output_tmp_path
#
#
# @app.post("/admin/audio-transcribe-part", response_model=TranscriptionPartResponse)
# async def transcribe_audio_part(
#         file: UploadFile = File(...),
#         part_id: int = Form(...),
#         subject: Optional[str] = Form(None),
#         language: Optional[str] = Form(None)
# ):
#     language = language or 'ru'
#     subject = subject or 'Brak przedmiotu'
#     filename = file.filename.lower()
#
#     if not is_allowed_file(filename):
#         raise HTTPException(status_code=400, detail="Nieobsługiwany format pliku audio.")
#
#     audio_bytes = await file.read()
#     if len(audio_bytes) > MAX_FILE_SIZE:
#         raise HTTPException(status_code=400, detail="Plik audio jest zbyt duży. Maksymalnie 100 MB.")
#
#     try:
#         wav_path = convert_to_wav(audio_bytes)
#
#         with wave.open(wav_path, "rb") as wav_file:
#             frames = wav_file.getnframes()
#             rate = wav_file.getframerate()
#             duration = frames / float(rate)
#
#         if duration > MAX_AUDIO_DURATION:
#             os.remove(wav_path)
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Plik audio jest za długi: {duration:.2f} sekund. Maksymalnie 30 minut."
#             )
#
#         segments, info = whisper_model.transcribe(
#             wav_path,
#             beam_size=5,
#             language=language,
#             temperature=0.2
#         )
#         transcription = " ".join(segment.text.strip() for segment in segments).strip()
#         os.remove(wav_path)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Błąd serwera: {str(e)}")
#
#     return TranscriptionPartResponse(
#         part_id=int(part_id),
#         transcription=str(transcription),
#         language=str(info.language),
#         language_probability=float(round(info.language_probability, 2)) if info.language_probability is not None else None,
#         subject=str(subject) if subject else None
#     )


def split_text_into_sentences(text: str, language: str = 'ru') -> list[str]:
    nlp = nlp_models.get(language)
    if nlp:
        try:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if sentences:
                return sentences
        except Exception as e:
            logger.warning(f"SpaCy failed, fallback to regex: {e}")
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
async def subtopics_generate(data: SubtopicsGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_subtopics_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return SubtopicsGenerator(**old_data)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        response = await request_ai(old_data['prompt'], old_data, request)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['subtopics'] = parse_subtopics_response(old_data['subtopics'], response, old_data['errors'])
        new_data['errors'] = old_data['errors']

        logger.info(previous_errors)
        logger.info(new_data['errors'])

        if sorted(new_data['subtopics']) == sorted(old_data['subtopics']) and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        new_data['attempt'] = new_data['attempt'] + 1

        return SubtopicsGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return SubtopicsGenerator(**old_data)

@app.post("/admin/task-generate")
async def task_generate(data: TaskGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_task_response,
            parse_note_response,
            parse_task_output_subtopics_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return TaskGenerator(**old_data)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        response = await request_ai(old_data['prompt'], old_data, request)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['text'] = parse_task_response(old_data['text'], response, old_data['errors'])
        new_data['note'] = parse_note_response(old_data['note'], response, old_data['errors'])
        new_data['outputSubtopics'] = parse_task_output_subtopics_response(old_data['outputSubtopics'], old_data['subtopics'],
                                                                       response, old_data['errors'])
        new_data['errors'] = old_data['errors']

        if new_data['text'] == old_data['text'] and new_data['note'] == old_data['note'] and sorted(new_data['outputSubtopics']) == sorted(old_data['outputSubtopics']) and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        if new_data['text'] != "" and new_data['note'] != "" and len(new_data['outputSubtopics']) != 0:
            new_data['changed'] = "false"
            return TaskGenerator(**new_data)

        new_data['attempt'] = new_data['attempt'] + 1

        return TaskGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return TaskGenerator(**old_data)

@app.post("/admin/words-generate")
async def words_generate(data: WordsGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_words_output_text_response,
            parse_output_words_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return WordsGenerator(**old_data)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        response = await request_ai(old_data['prompt'], old_data, request)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['outputText'] = parse_words_output_text_response(old_data['outputText'], response, old_data['errors'])
        new_data['outputWords'] = parse_output_words_response(old_data['outputWords'], old_data['words'],
                                                                       response, old_data['errors'])
        new_data['errors'] = old_data['errors']

        if new_data['outputText'] == old_data['outputText'] and sorted(new_data['outputWords']) == sorted(old_data['outputWords']) and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        if new_data['outputText'] != "":
            new_data['changed'] = "false"
            return WordsGenerator(**new_data)

        new_data['attempt'] = new_data['attempt'] + 1

        return WordsGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return WordsGenerator(**old_data)

@app.post("/admin/interactive-task-generate")
async def interactive_task_generate(data: InteractiveTaskGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_interactive_task_text_response,
            parse_interactive_task_translate_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return InteractiveTaskGenerator(**old_data)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        response = await request_ai(old_data['prompt'], old_data, request)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['text'] = parse_interactive_task_text_response(old_data['text'], response, old_data['errors'])
        new_data['translate'] = parse_interactive_task_translate_response(old_data['translate'], response, old_data['errors'])
        new_data['errors'] = old_data['errors']

        if new_data['text'] == old_data['text'] and new_data['translate'] == old_data['translate'] and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        if new_data['text'] != "" and new_data['translate'] != "":
            new_data['changed'] = "false"
            return InteractiveTaskGenerator(**new_data)

        new_data['attempt'] = new_data['attempt'] + 1

        return InteractiveTaskGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return InteractiveTaskGenerator(**old_data)

@app.post("/admin/questions-task-generate")
async def questions_task_generate(data: QuestionsTaskGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_questions_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return QuestionsTaskGenerator(**old_data)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        response = await request_ai(old_data['prompt'], old_data, request)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['questions'] = parse_questions_response(old_data['questions'], response, old_data['errors'])
        new_data['errors'] = old_data['errors']

        if sorted(new_data['questions']) == sorted(old_data['questions']) and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        if len(new_data['questions']) != 0:
            new_data['changed'] = "false"
            return QuestionsTaskGenerator(**new_data)

        new_data['attempt'] = new_data['attempt'] + 1

        return QuestionsTaskGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return QuestionsTaskGenerator(**old_data)

@app.post("/admin/solution-generate")
async def solution_generate(data: SolutionGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_solution_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return SolutionGenerator(**old_data)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        response = await request_ai(old_data['prompt'], old_data, request)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['solution'] = parse_solution_response(old_data['solution'], response, old_data['errors'])
        new_data['errors'] = old_data['errors']

        if new_data['solution'] == old_data['solution'] and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        if new_data['solution'] != "":
            new_data['changed'] = "false"
            return SolutionGenerator(**new_data)

        new_data['attempt'] = new_data['attempt'] + 1

        return SolutionGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return SolutionGenerator(**old_data)

@app.post("/admin/options-generate")
async def options_generate(data: OptionsGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_options_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return OptionsGenerator(**old_data)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        response = await request_ai(old_data['prompt'], old_data, request)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        result = parse_options_response(old_data, response, old_data['errors'])

        new_data['options'] = result['options']
        new_data['correctOptionIndex'] = result['correctOptionIndex']
        new_data['explanations'] = result['explanations']
        new_data['errors'] = old_data['errors']

        if new_data['correctOptionIndex'] == old_data['correctOptionIndex'] and sorted(new_data['explanations']) == sorted(new_data['explanations']) and sorted(new_data['options']) == sorted(old_data['options']) and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        if len(new_data['options']) == 4 and len(new_data['explanations']) == 4:
            new_data['changed'] = "false"
            return OptionsGenerator(**new_data)

        new_data['attempt'] = new_data['attempt'] + 1

        return OptionsGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return OptionsGenerator(**old_data)

@app.post("/admin/problems-generate")
async def problems_generate(data: ProblemsGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_subtopics_response,
            parse_output_subtopics_response,
            parse_explanation_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return ProblemsGenerator(**old_data)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        response = await request_ai(old_data['prompt'], old_data, request)

        logger.info(f"Response: {response}")

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        result = parse_subtopics_response(old_data['outputSubtopics'], response, old_data['errors'], "Procent opanowania")
        result = parse_output_subtopics_response(old_data['outputSubtopics'], result, old_data['subtopics'], old_data['errors'])
        new_data['outputSubtopics'] = result
        new_data['explanation'] = parse_explanation_response(old_data['explanation'], response, old_data['errors'])
        new_data['errors'] = old_data['errors']

        if new_data['explanation'] == old_data['explanation'] and sorted(new_data['outputSubtopics']) == sorted(old_data['outputSubtopics']) and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        if len(new_data['outputSubtopics']) != 0 and new_data['explanation'] != "":
            new_data['changed'] = "false"
            return ProblemsGenerator(**new_data)

        new_data['attempt'] = new_data['attempt'] + 1

        return ProblemsGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return ProblemsGenerator(**old_data)

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