# Берём официальный образ PyTorch с CUDA
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Локаль для pip
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Рабочая директория
WORKDIR /app

# Копируем зависимости
COPY requirements.txt /app/

# Устанавливаем зависимости Python и всегда последнюю версию g4f
RUN python3 -m pip install --no-cache-dir -r requirements.txt \
    && python3 -m pip install --upgrade g4f

# Загружаем модели spacy один раз при сборке
RUN python3 -m spacy download pl_core_news_sm \
    && python3 -m spacy download ru_core_news_sm \
    && python3 -m spacy download en_core_web_sm

# Копируем весь проект
COPY . /app

# Переменные окружения
ENV PORT=4200

# Экспонируем порт
EXPOSE 4200

# CMD для запуска FastAPI через uvicorn
CMD ["python3.11", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4200", "--timeout-keep-alive", "900", "--timeout-graceful-shutdown", "900"]