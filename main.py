from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body, Request
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from dotenv import load_dotenv
import os
import re
from fastapi.responses import FileResponse
import uuid
import boto3
import urllib.parse
from io import BytesIO
import copy
import json
import aiohttp
import logging
import sys
import asyncio
import random
from openai import OpenAI
from difflib import SequenceMatcher
from fastapi.responses import StreamingResponse
from collections import Counter
from azure.core.credentials import AzureKeyCredential
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig, ResultReason

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

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

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

MAX_TEXT_LENGTH = 5000

tts_semaphore = asyncio.Semaphore(2)

speech_key = os.getenv("AZURE_SPEECH_KEY")
speech_region = os.getenv("AZURE_SPEECH_REGION")

ALLOWED_EXTENSIONS = {
    '.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.wma', '.opus',
    '.amr', '.aiff', '.alac', '.pcm', '.webm', '.mp4', '.3gp', '.caf'
}

MAX_AUDIO_DURATION = 900

MAX_ATTEMPTS = 2

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

async def request_ai(
        prompt: str,
        data: Dict[str, Any],
        request: Request,
        max_retries: int = 1,
        stream: bool = False,
        model: str = "deepseek-chat",
        web_search = False
) -> Optional[str]:
    prompt_filled = fill_placeholders(prompt, data)

    logger.info(prompt_filled)

    for attempt in range(max_retries + 1):
        logger.info(f"[Attempt {attempt + 1}]")

        try:
            if stream:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "user", "content": prompt_filled}
                            ],
                            temperature=0,
                            stream=True,
                            web_search_options=web_search,
                            max_tokens=16384
                        )
                    ),
                    timeout=900
                )

                chunks = []
                current_segment = ""
                chunk_count = 0

                for chunk in response:
                    chunk_count += 1

                    try:
                        if (chunk and chunk.choices and chunk.choices[0].delta.content):
                            text = chunk.choices[0].delta.content
                            if text:
                                chunks.append(text)
                                current_segment += text

                                if len(current_segment) > 80 and (
                                        '\n' in current_segment or '. ' in current_segment[-20:]):
                                    logger.info(current_segment.strip())
                                    current_segment = ""
                    except AttributeError:
                        continue

                if current_segment.strip():
                    logger.info(current_segment.strip())

                content = "".join(chunks).strip()
            else:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "user", "content": prompt_filled}
                            ],
                            temperature=0,
                            web_search_options=web_search,
                            stream=False,
                            max_tokens=16384
                        )
                    ),
                    timeout=900
                )

                if response.choices and response.choices[0].message.content:
                    content = response.choices[0].message.content.strip()
                else:
                    content = ""

                if content:
                    logger.info(f"Response: {content}")

            if content:
                if len(content) < 10:
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                    continue

                if not content or content.isspace():
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                    continue

                if content.endswith(('...', '--', '[', '{', '(')):
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                    continue

                return content

            if attempt < max_retries:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)

        except Exception as e:
            logger.error(f"Error: {e}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)

    logger.error(f"All {max_retries + 1} attempts failed")
    return None

class PromptImageRequest(BaseModel):
    prompt: str
    data: Optional[dict] = {}
    image_base64: Optional[str] = None

class PromptRequest(BaseModel):
    prompt: str

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
    information: str
    accounts: str
    balance: str
    subtopics: List[List]
    attempt: int
    prompt: str
    errors: List[str]

class SubtopicsStatusGenerator(BaseModel):
    changed: str
    subject: str
    section: str
    topic: str
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
    information: str
    accounts: str
    balance: str
    subtopics: List[str]
    attempt: int
    prompt: str
    errors: List[str]

class SolutionGenerator(BaseModel):
    changed: str
    subject: str
    section: str
    topic: str
    text: str
    solution: str
    information: str
    accounts: str
    balance: str
    attempt: int
    prompt: str
    errors: List[str]

class FrequencyGenerator(BaseModel):
    changed: str
    subject: str
    section: str
    topic: str
    literature: str
    information: str
    accounts: str
    balance: str
    content: str
    frequency: int
    attempt: int
    prompt: str
    errors: List[str]

class ChronologyGenerator(BaseModel):
    changed: str
    subject: str
    section: str
    topic: str
    literature: str
    information: str
    accounts: str
    balance: str
    content: str
    subtopics: List[List]
    outputSubtopics: List[List]
    attempt: int
    prompt: str
    errors: List[str]

class TaskGenerator(BaseModel):
    changed: str
    subject: str
    section: str
    topic: str
    literature: str
    information: str
    accounts: str
    balance: str
    subtopics: List[str]
    outputSubtopics: List[str]
    threshold: int
    text: str
    attempt: int
    prompt: str
    errors: List[str]

class WritingGenerator(BaseModel):
    changed: str
    subject: str
    section: str
    topic: str
    literature: str
    information: str
    accounts: str
    balance: str
    text: str
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
    words: List[str]
    outputWords: List[str]
    text: str
    translate: str
    attempt: int
    prompt: str
    errors: List[str]

class OptionsGenerator(BaseModel):
    changed: str
    text: str
    solution: str
    information: str
    accounts: str
    balance: str
    options: List[str]
    correctOptionIndex: int
    attempt: int
    prompt: str
    errors: List[str]
    subtopics: List[str]
    randomOption: int

class ProblemsGenerator(BaseModel):
    changed: str
    text: str
    chat: str
    type: str
    explanation: str
    solution: str
    information: str
    accounts: str
    balance: str
    options: List[str]
    subtopics: List[str]
    correctOption: str
    outputSubtopics: List[list]
    subject: str
    section: str
    topic: str
    userSolution: str
    options: List[str]
    correctOption: str
    userOption: str
    attempt: int
    prompt: str
    errors: List[str]

class ChatGenerator(BaseModel):
    explanation: str
    changed: str
    text: str
    solution: str
    subject: str
    section: str
    topic: str
    information: str
    accounts: str
    balance: str
    userSolution: str
    options: List[str]
    correctOption: str
    userOption: str
    chat: str
    chatFinished: bool
    subtopics: List[str]
    attempt: int
    prompt: str
    errors: List[str]

class LiteratureGenerator(BaseModel):
    changed: str
    name: str
    note: str
    attempt: int
    prompt: str
    errors: List[str]

class VocabluaryGenerator(BaseModel):
    changed: str
    words: List[List]
    outputWords: List[str]
    outputText: str
    attempt: int
    prompt: str
    errors: List[str]

class VocabluaryGuideGenerator(BaseModel):
    changed: str
    text: str
    translate: str
    attempt: int
    prompt: str
    errors: List[str]

class WordsGenerator(BaseModel):
    changed: str
    subject: str
    section: str
    topic: str
    type: str
    difficulty: str
    information: str
    words: List[List]
    attempt: int
    prompt: str
    errors: List[str]

class PromptImageRequest(BaseModel):
    prompt: str
    data: Optional[dict] = {}

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
async def generate_tts(data: TTSRequest):
    if len(data.text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Text too long (max {MAX_TEXT_LENGTH} chars)")

    async with tts_semaphore:
        try:
            if not speech_key or not speech_region:
                raise HTTPException(status_code=500, detail="Azure TTS credentials not found")

            speech_config = SpeechConfig(subscription=speech_key, region=speech_region)
            language_voice_map = {
                "en": "en-US-AriaNeural",
                "pl": "pl-PL-MarekNeural",
                "ru": "ru-RU-DariyaNeural"
            }
            voice = language_voice_map.get(data.language.lower(), "en-US-AriaNeural")
            speech_config.speech_synthesis_voice_name = voice

            mp3_path = f"/tmp/tts_{uuid.uuid4()}.mp3"
            audio_config = AudioConfig(filename=mp3_path)
            synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            result = synthesizer.speak_text_async(data.text).get()

            logger.info(f"TTS RESULT REASON: {result.reason}")

            if result.reason == ResultReason.Canceled:
                cancellation = result.cancellation_details

                logger.error(f"CANCEL REASON: {cancellation.reason}")

                if cancellation.error_details:
                    logger.error(f"AZURE ERROR DETAILS: {cancellation.error_details}")

                raise HTTPException(
                    status_code=500,
                    detail=f"Azure TTS canceled: {cancellation.error_details}"
                )

            elif result.reason != ResultReason.SynthesizingAudioCompleted:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected TTS result: {result.reason}"
                )

            with open(mp3_path, "rb") as f:
                mp3_bytes = BytesIO(f.read())
                mp3_bytes.seek(0)

            filename = f"tts_{data.id}_{data.part_id}_{uuid.uuid4()}.mp3"
            await asyncio.to_thread(
                s3.upload_fileobj,
                mp3_bytes,
                BUCKET_NAME,
                filename,
                ExtraArgs={"ContentType": "audio/mpeg"}
            )

            public_url = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{urllib.parse.quote(filename)}"
            return {"url": public_url}

        except Exception as e:
            logger.error(f"TTS ERROR: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/subtopics-generate")
async def subtopics_generate(data: SubtopicsGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_subtopics_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return SubtopicsGenerator(**old_data)

        response = await request_ai(old_data['prompt'], old_data, request, stream=False, model="deepseek-reasoner", web_search=True)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['subtopics'] = parse_subtopics_response(old_data['subtopics'], response, old_data['errors'])
        new_data['errors'] = old_data['errors']
        new_data['attempt'] = new_data['attempt'] + 1

        if sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        return SubtopicsGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return SubtopicsGenerator(**old_data)

@app.post("/admin/subtopics-status-generate")
async def subtopics_status_generate(data: SubtopicsStatusGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_subtopics_status_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return SubtopicsStatusGenerator(**old_data)

        response = await request_ai(old_data['prompt'], old_data, request, stream=False, model="deepseek-reasoner")

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['subtopics'] = parse_subtopics_status_response(old_data['subtopics'], response, old_data['errors'])
        new_data['errors'] = old_data['errors']

        new_data['attempt'] = new_data['attempt'] + 1

        if sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        return SubtopicsStatusGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return SubtopicsStatusGenerator(**old_data)

@app.post("/admin/topic-expansion-generate")
async def topic_expansion_generate(data: TopicExpansionGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_note_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return TopicExpansionGenerator(**old_data)

        response = await request_ai(old_data['prompt'], old_data, request, stream=False, web_search=True)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['note'] = parse_note_response(old_data['note'], response, old_data['errors'])
        new_data['errors'] = old_data['errors']

        logger.info(previous_errors)
        logger.info(new_data['errors'])
        new_data['attempt'] = new_data['attempt'] + 1

        if new_data['note'] != "" and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        return TopicExpansionGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return TopicExpansionGenerator(**old_data)

@app.post("/admin/solution-generate")
async def solution_generate(data: SolutionGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_solution_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return SolutionGenerator(**old_data)

        response = await request_ai(old_data['prompt'], old_data, request, stream=False)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['solution'] = parse_solution_response(old_data['solution'], response, old_data['errors'])
        new_data['errors'] = old_data['errors']

        logger.info(previous_errors)
        logger.info(new_data['errors'])
        new_data['attempt'] = new_data['attempt'] + 1

        if new_data['solution'] != "" and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        return SolutionGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return SolutionGenerator(**old_data)

@app.post("/admin/chronology-generate")
async def chronology_generate(data: ChronologyGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_subtopics_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return ChronologyGenerator(**old_data)

        response = await request_ai(old_data['prompt'], old_data, request, stream=False, model="deepseek-reasoner")

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['outputSubtopics'] = parse_subtopics_response(
            old_data['outputSubtopics'],
            response,
            old_data['errors'],
            "Numer Porządkowy",
            "subtopicsStart:",
            "subtopicsEnd:")
        new_data['errors'] = old_data['errors']

        logger.info(previous_errors)
        logger.info(new_data['errors'])
        new_data['attempt'] = new_data['attempt'] + 1

        if sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        return ChronologyGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return ChronologyGenerator(**old_data)

@app.post("/admin/frequency-generate")
async def frequency_generate(data: FrequencyGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_frequency_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return FrequencyGenerator(**old_data)

        response = await request_ai(old_data['prompt'], old_data, request, stream=False, model="deepseek-reasoner")

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['frequency'] = parse_frequency_response(old_data['frequency'], response, old_data['errors'])
        new_data['errors'] = old_data['errors']

        logger.info(previous_errors)
        logger.info(new_data['errors'])
        new_data['attempt'] = new_data['attempt'] + 1

        if new_data['frequency'] != 0 and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        return FrequencyGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return FrequencyGenerator(**old_data)

@app.post("/admin/task-generate")
async def task_generate(data: TaskGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_task_response,
            parse_output_subtopics_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return TaskGenerator(**old_data)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        logger.info(old_data['literature'])

        response = await request_ai(old_data['prompt'], old_data, request, stream=False)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['text'] = parse_task_response(old_data['text'], response, old_data['errors'])
        new_data['outputSubtopics'] = parse_output_subtopics_response(old_data['outputSubtopics'], old_data['subtopics'],
                                                                       response, old_data['errors'])
        new_data['errors'] = old_data['errors']
        new_data['attempt'] = new_data['attempt'] + 1

        if new_data['text'] != "" and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        return TaskGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return TaskGenerator(**old_data)

@app.post("/admin/writing-generate")
async def writing_generate(data: WritingGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_task_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return WritingGenerator(**old_data)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        logger.info(old_data['literature'])

        response = await request_ai(old_data['prompt'], old_data, request, stream=False)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['text'] = parse_task_response(old_data['text'], response, old_data['errors'])
        new_data['errors'] = old_data['errors']
        new_data['attempt'] = new_data['attempt'] + 1

        if new_data['text'] != "" and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        return WritingGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return WritingGenerator(**old_data)

@app.post("/admin/vocabluary-generate")
async def vocabluary_generate(data: VocabluaryGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_words_output_text_response,
            parse_output_words_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return VocabluaryGenerator(**old_data)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        response = await request_ai(old_data['prompt'], old_data, request, stream=False)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['outputText'] = parse_words_output_text_response(old_data['outputText'], response, old_data['errors'])
        new_data['outputWords'] = parse_output_words_response(old_data['outputWords'], old_data['words'],
                                                                       response, old_data['errors'])
        new_data['errors'] = old_data['errors']
        new_data['attempt'] = new_data['attempt'] + 1

        if new_data['outputText'] != "" and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        return VocabluaryGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return VocabluaryGenerator(**old_data)

@app.post("/admin/vocabluary-guide-generate")
async def vocabluary_guide_generate(data: VocabluaryGuideGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_translate_response
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return VocabluaryGuideGenerator(**old_data)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        response = await request_ai(old_data['prompt'], old_data, request, stream=False, model="deepseek-reasoner")

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['translate'] = parse_translate_response(old_data['translate'], response, old_data['errors'])
        new_data['errors'] = old_data['errors']
        new_data['attempt'] = new_data['attempt'] + 1

        if new_data['translate'] != "" and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        return VocabluaryGuideGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return VocabluaryGuideGenerator(**old_data)

@app.post("/admin/interactive-task-generate")
async def interactive_task_generate(data: InteractiveTaskGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_interactive_task_text_response,
            parse_interactive_task_translate_response,
            parse_output_words_response,
            remove_markdown
        )

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return InteractiveTaskGenerator(**old_data)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        response = await request_ai(old_data['prompt'], old_data, request, stream=False)
        if response:
            response = remove_markdown(response)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        new_data['text'] = parse_interactive_task_text_response(old_data['text'], response, old_data['errors'])
        new_data['translate'] = parse_interactive_task_translate_response(old_data['translate'], response, old_data['errors'])
        new_data['outputWords'] = parse_output_words_response(old_data['outputWords'], old_data['words'], response, old_data['errors'], False)
        new_data['errors'] = old_data['errors']
        new_data['attempt'] = new_data['attempt'] + 1

        if new_data['text'] != "" and new_data['translate'] != "" and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        return InteractiveTaskGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return InteractiveTaskGenerator(**old_data)

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

        response = await request_ai(old_data['prompt'], old_data, request, stream=False, model="deepseek-reasoner")

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        result = parse_options_response(old_data, response, old_data['errors'])

        new_data['options'] = result['options']
        new_data['correctOptionIndex'] = new_data['randomOption'] - 1
        new_data['attempt'] = new_data['attempt'] + 1

        if sorted(old_data['options']) == sorted(new_data['options']):
            new_data['changed'] = "false"

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
            parse_output_subtopics_response_filtered,
            parse_explanation_response,
            parse_writing_explanation
        )

        user_solution = old_data.get('userSolution', '')
        subtopics_list = old_data.get('subtopics', [])

        if old_data['changed'] == "false" or old_data['attempt'] > MAX_ATTEMPTS:
            return ProblemsGenerator(**old_data)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        response = await request_ai(old_data['prompt'], old_data, request, stream=False)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])
        result = parse_subtopics_response(old_data['outputSubtopics'], response, old_data['errors'],
                                          "Procent opanowania")
        result = parse_output_subtopics_response_filtered(old_data['outputSubtopics'], result, old_data['subtopics'],
                                                          old_data['errors'])
        new_data['outputSubtopics'] = result

        if new_data['type'] == "Writing":
            new_data['explanation'] = parse_writing_explanation(
                old_data['explanation'],
                response,
                old_data['errors']
            )
        else:
            new_data['explanation'] = parse_explanation_response(
                old_data['explanation'],
                response,
                old_data['errors'],
                new_data['outputSubtopics'],
                new_data['correctOption'],
                new_data['userOption'],
                new_data['topic'],
                new_data['type']
            )

        new_data['errors'] = old_data['errors']
        new_data['attempt'] = new_data['attempt'] + 1

        if len(new_data['outputSubtopics']) != 0 and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        return ProblemsGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return ProblemsGenerator(**old_data)

def strip_chat_tags(text: str) -> str:
    if text is None:
        return text

    text = re.sub(r'</?chat>', '', text, flags=re.IGNORECASE)

    text = re.sub(r'\n\s*\n', '\n\n', text)

    return text.strip()

def ensure_chat_tags(text: str) -> str:
    if text is None:
        return text

    text = text.strip()

    if not text.startswith("<chat>"):
        text = "<chat>\n" + text

    if not text.endswith("</chat>"):
        text = text + "\n</chat>"

    return text

@app.post("/admin/chat-generate")
async def chat_generate(data: ChatGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_chat_response
        )

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        response = await request_ai(old_data['prompt'], old_data, request, stream=False)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        if response:
            response = strip_chat_tags(response)
            response = ensure_chat_tags(response)

        if "[AI_QUESTION]" not in response:
            old_data['errors'] = ["Nie ma marker [AI_QUESTION] - on jest WYMAGANY!"]
            response = await request_ai(old_data['prompt'], old_data, request, stream=False)

            if await request.is_disconnected():
                raise HTTPException(status_code=499, detail="Client disconnected")

            if response:
                response = strip_chat_tags(response)
                response = ensure_chat_tags(response)

        parsed_chat = parse_chat_response(old_data['chat'], response, old_data['errors'])
        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])

        new_data['chat'] = parsed_chat
        new_data['errors'] = old_data['errors']
        new_data['attempt'] = new_data['attempt'] + 1

        if new_data['chat'] != "" and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        return ChatGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return ChatGenerator(**old_data)

@app.post("/admin/literature-generate")
async def literature_generate(data: LiteratureGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    try:
        from ai_generator import (
            parse_literature_response
        )

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        response = await request_ai(old_data['prompt'], old_data, request, stream=False, model="deepseek-reasoner", web_search=True)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        new_data = copy.deepcopy(old_data)
        previous_errors = copy.deepcopy(old_data['errors'])

        new_data['note'] = parse_literature_response(old_data['note'], response, old_data['errors'])

        new_data['errors'] = old_data['errors']
        new_data['attempt'] = new_data['attempt'] + 1

        if new_data['note'] != "" and sorted(previous_errors) == sorted(new_data['errors']):
            new_data['changed'] = "false"

        return LiteratureGenerator(**new_data)
    except RuntimeError as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return LiteratureGenerator(**old_data)

@app.post("/admin/words-generate")
async def words_generate(data: WordsGenerator, request: Request):
    old_data = copy.deepcopy(data.dict())

    target_generations = 5
    min_frequency = 30
    accumulated_lists = []

    try:
        from ai_generator import parse_words_response

        response = await request_ai(old_data['prompt'], old_data, request, stream=False, model="deepseek-chat")
        new_words = parse_words_response([], response, old_data['errors'])

        new_data = copy.deepcopy(old_data)
        new_data['words'] = new_words
        new_data['changed'] = "false"
        return WordsGenerator(**new_data)
    except Exception as e:
        old_data['errors'].append(str(e))
        old_data['changed'] = 'true'
        old_data['attempt'] += 1
        return WordsGenerator(**old_data)

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