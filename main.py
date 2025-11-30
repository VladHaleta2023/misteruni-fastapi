from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body, Request
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from dotenv import load_dotenv
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
from openai import OpenAI
from difflib import SequenceMatcher
from collections import Counter

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
api_key = os.environ.get("DEEPSEEK_API_KEY")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

app = FastAPI()

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

BUCKET_NAME = os.getenv("AWS_BUCKET")
REGION = os.getenv("AWS_REGION")

MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    '.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.wma', '.opus',
    '.amr', '.aiff', '.alac', '.pcm', '.webm', '.mp4', '.3gp', '.caf'
}

MAX_AUDIO_DURATION = 900

MAX_ATTEMPTS = 1

def calculate_word_similarity(student_answer, correct_answer):
    student_words = set(re.findall(r'\w+', student_answer.lower()))
    correct_words = set(re.findall(r'\w+', correct_answer.lower()))

    if not correct_words:
        return 0

    common_words = student_words.intersection(correct_words)
    similarity = len(common_words) / len(correct_words)

    return similarity

def calculate_sequence_similarity(student_answer, correct_answer):
    return SequenceMatcher(None, student_answer.lower(), correct_answer.lower()).ratio()

def extract_key_phrases(text, phrase_length=3):
    words = re.findall(r'\w+', text.lower())
    phrases = []

    # Создаем фразы из последовательных слов
    for i in range(len(words) - phrase_length + 1):
        phrase = " ".join(words[i:i + phrase_length])
        phrases.append(phrase)

    return phrases

def check_key_phrases(student_answer, correct_answer, min_phrase_length=3):
    student_lower = student_answer.lower()
    correct_phrases = extract_key_phrases(correct_answer, min_phrase_length)

    matches = 0
    for phrase in correct_phrases:
        if phrase in student_lower:
            matches += 1

    return matches

def is_copy_combined(student_answer, correct_answer,
                     word_threshold=0.7,
                     sequence_threshold=0.75,
                     key_phrases_threshold=2):
    if not student_answer or not correct_answer:
        return False

    word_similarity = calculate_word_similarity(student_answer, correct_answer)
    seq_similarity = calculate_sequence_similarity(student_answer, correct_answer)
    key_phrases_match = check_key_phrases(student_answer, correct_answer)

    return (word_similarity >= word_threshold and
            seq_similarity >= sequence_threshold and
            key_phrases_match >= key_phrases_threshold)

def fill_placeholders(prompt: str, data: Dict[str, Any]) -> str:
    def replacer(match: re.Match) -> str:
        key = match.group(1)
        value = data.get(key, "")
        if isinstance(value, list):
            if not value:
                return f"{key}Start:\n{value}\n{key}End:"
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
    logger.info("Model deepseek-chat started processing request...")
    logger.info(prompt_filled)

    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt_filled}],
                    temperature=0,
                    stream=False,
                )
            ),
            timeout=900
        )

        content = resp.choices[0].message.content.strip()
        logger.info("✅ Model deepseek-chat returned a valid result.")
        logger.info(content)
        return content

    except asyncio.TimeoutError:
        logger.error("⏳ Model deepseek-chat timed out after 900 seconds (15 minutes).")
        return None
    except Exception as e:
        logger.error(f"❌ Model deepseek-chat failed with error: {e}")
        return None

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
    literature: str
    subtopics: List[List]
    attempt: int
    prompt: str
    errors: List[str]

class TopicExpansionGenerator(BaseModel):
    changed: str
    subject: str
    section: str
    topic: str
    literature: str
    note: str
    frequency: int
    subtopics: List[str]
    attempt: int
    prompt: str
    errors: List[str]

class TaskGenerator(BaseModel):
    changed: str
    subject: str
    section: str
    topic: str
    mode: str
    literature: str
    subtopics: List[List]
    outputSubtopics: List[str]
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
    difficulty: str
    subtopics: List[List]
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
    subtopics: List[str]

class OptionsGenerator(BaseModel):
    changed: str
    text: str
    solution: str
    options: List[str]
    explanations: List[str]
    attempt: int
    prompt: str
    errors: List[str]
    subtopics: List[str]
    random1: int
    random2: int

class ProblemsGenerator(BaseModel):
    changed: str
    text: str
    explanation: str
    solution: str
    options: List[str]
    subtopics: List[str]
    correctOption: str
    outputSubtopics: List[list]
    subject: str
    section: str
    topic: str
    userSolution: str
    attempt: int
    prompt: str
    errors: List[str]

class WordsGenerator(BaseModel):
    changed: str
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

COMMON_ABBREVIATIONS = {
    "pl": [
        "np.", "itp.", "tj.", "dr.", "mgr.", "prof.", "ul.", "al.", "św.",
        "r.", "god.", "p.", "nr.", "ok.", "m.in.", "cdn.", "tzn."
    ],
    "en": [
        "mr.", "mrs.", "ms.", "dr.", "prof.", "inc.", "ltd.", "jr.", "sr.",
        "st.", "vs.", "u.s.", "u.k.", "e.g.", "i.e.", "etc.", "fig."
    ],
}

def split_text_into_sentences(text: str, language: str = "en") -> list[str]:
    if not text or not isinstance(text, str):
        return []

    abbreviations = COMMON_ABBREVIATIONS.get(language.lower(), [])
    abbrev_pattern = re.compile(
        r"\b(" + "|".join([re.escape(a) for a in abbreviations]) + r")",
        re.IGNORECASE
    )

    text_protected = abbrev_pattern.sub(lambda m: m.group(1).replace(".", "§"), text)

    text_protected = re.sub(r"(\d)\.(\d)", r"\1§\2", text_protected)

    parts = re.split(
        r"(?<=[.!?…])\s+(?=[\"“”'«»„\(]*[A-ZА-ЯŁŚŹŻĆŃ])",
        text_protected
    )

    sentences = []
    for p in parts:
        s = p.strip().replace("§", ".")
        if s:
            if sentences and (
                len(s) < 3 or (s and s[0].islower())
            ):
                sentences[-1] += " " + s
            else:
                sentences.append(s)

    return sentences

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

        response = await request_ai(old_data['prompt'], old_data, request)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['subtopics'] = parse_subtopics_response(old_data['subtopics'], response, old_data['errors'])
        new_data['errors'] = old_data['errors']

        if sorted(new_data['subtopics']) == sorted(old_data['subtopics']) and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        new_data['attempt'] = new_data['attempt'] + 1

        return SubtopicsGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return SubtopicsGenerator(**old_data)

@app.post("/admin/topic-expansion-generate")
async def topic_expansion_generate(data: TopicExpansionGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_note_response,
            parse_frequency_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return TopicExpansionGenerator(**old_data)

        response = await request_ai(old_data['prompt'], old_data, request)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['frequency'] = parse_frequency_response(old_data['frequency'], response, old_data['errors'])
        new_data['note'] = parse_note_response(old_data['note'], response, old_data['errors'])
        new_data['errors'] = old_data['errors']

        logger.info(previous_errors)
        logger.info(new_data['errors'])

        if new_data['note'] == old_data['note'] and new_data['frequency'] == old_data['frequency'] and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        if new_data['note'] != "" and new_data['frequency'] != 0:
            new_data['changed'] = "false"
            return TopicExpansionGenerator(**new_data)

        new_data['attempt'] = new_data['attempt'] + 1

        return TopicExpansionGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return TopicExpansionGenerator(**old_data)

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

        logger.info(old_data['literature'])

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

        if new_data['text'] == old_data['text'] and sorted(new_data['outputSubtopics']) == sorted(old_data['outputSubtopics']) and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

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

        #if len(new_data['questions']) != 0:
        #    new_data['changed'] = "false"
        #    return QuestionsTaskGenerator(**new_data)

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
        new_data['explanations'] = result['explanations']

        if (sorted(new_data['explanations']) == sorted(old_data['explanations']) and
                sorted(new_data['options']) == sorted(old_data['options']) and
                sorted(previous_errors) == sorted(new_data['errors'])):
            new_data['changed'] = "false"

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

        if is_copy_combined(old_data['userSolution'], old_data['correctOption']):
            new_data = copy.deepcopy(old_data)

            output_subtopics = []
            for subtopic in old_data['subtopics']:
                output_subtopics.append([subtopic, 85])

            new_data['outputSubtopics'] = output_subtopics
            new_data['explanation'] = "Odpowiedź została uznana za kopię poprawnego rozwiązania; zgodnie z zasadami oceniania przyznano karę −85%."
            new_data['changed'] = "false"

            return ProblemsGenerator(**new_data)

        response = await request_ai(old_data['prompt'], old_data, request)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        result = parse_subtopics_response(old_data['outputSubtopics'], response, old_data['errors'],
                                          "Procent opanowania")
        result = parse_output_subtopics_response(old_data['outputSubtopics'], result, old_data['subtopics'],
                                                 old_data['errors'])
        new_data['outputSubtopics'] = result
        new_data['explanation'] = parse_explanation_response(old_data['explanation'], response, old_data['errors'])
        new_data['errors'] = old_data['errors']

        if new_data['explanation'] == old_data['explanation'] and sorted(new_data['outputSubtopics']) == sorted(
                old_data['outputSubtopics']) and sorted(previous_errors) == sorted(new_data['errors']):
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