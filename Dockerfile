# Лёгкий Python образ для CPU
FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Локаль
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Рабочая директория
WORKDIR /app

# Копируем зависимости
COPY requirements.txt /app/

# Устанавливаем Python зависимости
RUN python3 -m pip install --no-cache-dir -r requirements.txt \
    && python3 -m pip install --upgrade g4f faster-whisper spacy

# Загружаем лёгкие модели spacy
RUN python3 -m spacy download pl_core_news_sm \
    && python3 -m spacy download ru_core_news_sm \
    && python3 -m spacy download en_core_web_sm

# Копируем проект
COPY . /app

# Экспонируем порт
EXPOSE 4200

# Запуск FastAPI с uvicorn, порт берём из переменной окружения Render
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-4200} --timeout-keep-alive 900 --timeout-graceful-shutdown 900"]