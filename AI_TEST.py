from g4f.client import Client
from typing import Optional, Dict, Any
import re

client = Client()

top_models = [
    "gpt-4.5", "gpt-4o", "gpt-4-turbo", "gpt-4",
    "claude-3-opus", "qwen-3-235b", "qwen-2.5-max",
    "deepseek-v3", "deepseek-r1-turbo", "qwen-2-72b",
    "gemini-2.5-pro", "wizardlm-2-8x22b", "llama-3.3-70b",
    "llama-3.2-90b", "llama-3.1-70b", "llama-4-maverick",
    "llama-3.1-405b", "mixtral-8x22b", "phi-4",
    "gemma-3-27b", "mistral-small-24b"
]

def request_ai(prompt: str) -> Optional[str]:
    for model in top_models:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=90,
                web_search=False
            )
            content = resp.choices[0].message.content.strip()
            if content:
                return content
        except Exception as e:
            print(f"Model {model} failed with error: {e}")
            continue
    return None

def metadata_fill_template(template: str, data: Dict[str, Any]) -> str:
    def replacer(match):
        key = match.group(1)
        value = data.get(key)

        if value is None:
            return ''

        if isinstance(value, list):
            return ', '.join(str(item) for item in value)
        else:
            return str(value)

    return re.sub(r'{(\w+)}', replacer, template)

