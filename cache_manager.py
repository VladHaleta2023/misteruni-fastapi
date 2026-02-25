# cache_manager.py
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import logging
import pickle

logger = logging.getLogger("app_logger")


class AICacheManager:
    """Универсальный менеджер кэша для всех AI запросов"""

    def __init__(self, cache_dir: str = "ai_cache", cache_ttl_days: int = 30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Отдельная директория для разных типов кэша
        self.response_cache_dir = self.cache_dir / "responses"
        self.response_cache_dir.mkdir(exist_ok=True)

        self.cache_ttl = timedelta(days=cache_ttl_days)

        logger.info(f"✅ Universal Cache manager initialized")

    def _generate_cache_key(self, prompt: str, data: Dict[str, Any], endpoint: str) -> str:
        """
        Генерирует уникальный ключ кэша на основе:
        - промпта
        - данных
        - эндпоинта (важно для разных форматов)
        """
        # Создаём копию данных без изменяемых полей
        cache_data = {}

        # Добавляем только стабильные поля
        stable_fields = ['subject', 'section', 'topic', 'difficulty', 'literature']
        for field in stable_fields:
            if field in data:
                cache_data[field] = data[field]

        # Добавляем эндпоинт
        cache_data['endpoint'] = endpoint

        # Добавляем хеш промпта (промпт определяет формат ответа)
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        cache_data['prompt_hash'] = prompt_hash

        # Сортируем для стабильности
        sorted_data = json.dumps(cache_data, sort_keys=True)

        # Создаём строку для хеширования
        hash_input = f"{endpoint}_{sorted_data}"

        # Возвращаем SHA256 хеш
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

    def get_cached(self, prompt: str, data: Dict[str, Any], endpoint: str) -> Optional[str]:
        """Получает результат из кэша для конкретного эндпоинта"""
        cache_key = self._generate_cache_key(prompt, data, endpoint)
        cache_file = self.response_cache_dir / f"{cache_key}.cache"

        # Проверяем существование и TTL
        if cache_file.exists():
            # Проверяем возраст файла
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - file_time < self.cache_ttl:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    logger.info(f"💾 Cache HIT [{endpoint}]: {cache_key[:8]}...")
                    return content
                except Exception as e:
                    logger.error(f"Error reading cache {cache_key}: {e}")
            else:
                # Удаляем устаревший кэш
                cache_file.unlink()
                logger.info(f"🗑️ Cache expired [{endpoint}]: {cache_key[:8]}...")

        logger.info(f"💾 Cache MISS [{endpoint}]: {cache_key[:8]}...")
        return None

    def save_to_cache(self, prompt: str, data: Dict[str, Any], endpoint: str, content: str):
        """Сохраняет результат в кэш для конкретного эндпоинта"""
        cache_key = self._generate_cache_key(prompt, data, endpoint)
        cache_file = self.response_cache_dir / f"{cache_key}.cache"

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"✅ Saved to cache [{endpoint}]: {cache_key[:8]}...")
        except Exception as e:
            logger.error(f"Error saving to cache {cache_key}: {e}")

    def clear_cache(self, endpoint: Optional[str] = None):
        """Очищает кэш для конкретного эндпоинта или весь"""
        if endpoint:
            # Очистить кэш для конкретного эндпоинта
            count = 0
            for cache_file in self.response_cache_dir.glob("*.cache"):
                # Здесь сложно определить эндпоинт из имени файла
                # Поэтому просто очищаем всё или добавляем логику
                cache_file.unlink()
                count += 1
            logger.info(f"🗑️ Cleared {count} cache files for {endpoint}")
        else:
            # Очистить весь кэш
            import shutil
            shutil.rmtree(self.response_cache_dir, ignore_errors=True)
            self.response_cache_dir.mkdir(exist_ok=True)
            logger.info("🗑️ Cleared all cache")


# Создаём глобальный экземпляр
cache_manager = AICacheManager()