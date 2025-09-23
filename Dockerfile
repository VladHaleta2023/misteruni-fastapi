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
# Сначала numpy, потом остальные
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir numpy==1.26.2 \
    && python3 -m pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --force-reinstall thinc==8.1.7

# Загружаем spaCy модели напрямую
RUN pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/pl_core_news_sm-3.6.0/pl_core_news_sm-3.6.0-py3-none-any.whl \
    https://github.com/explosion/spacy-models/releases/download/ru_core_news_sm-3.6.0/ru_core_news_sm-3.6.0-py3-none-any.whl \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl

# Копируем проект
COPY . /app

# Экспонируем порт
EXPOSE 4200

# Запуск FastAPI
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-4200} --timeout-keep-alive 900 --timeout-graceful-shutdown 900"]