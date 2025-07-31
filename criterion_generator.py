from typing import Dict, Any, Optional
import re
from configAI import client, top_models

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
        var_name = match.group(1)
        if not var_name.endswith('_input'):
            return match.group(0)

        key = var_name[:-6]
        if key not in data:
            return ''

        value = data[key]
        if isinstance(value, list):
            return '|'.join(sorted(str(item) for item in value))
        else:
            return str(value)

    return re.sub(r'{(\w+)}', replacer, template)

def extract_format_from_prompt(prompt: str) -> str:
    lines = [line.strip() for line in prompt.strip().splitlines() if line.strip()]
    for line in reversed(lines):
        if re.search(r'{\w+}', line):
            return line
    raise ValueError("Nie udało się znaleźć linii formatu z placeholderami w prompt")

def parse_output_by_format(format_str: str, output_str: str, data: Dict[str, Any]) -> Dict[str, Any]:
    keys = re.findall(r'{(\w+)}', format_str)
    parts = output_str.split('@@')
    if len(parts) != len(keys):
        raise ValueError(f"Liczba części output ({len(parts)}) nie zgadza się z liczbą zmiennych w formacie ({len(keys)})")
    for var_name, part in zip(keys, parts):
        if var_name.endswith('_output'):
            key = var_name[:-7]
        else:
            key = var_name
        if '|' in part:
            data[key] = part.split('|') if part else []
        else:
            data[key] = part
    return data

def normalize_value(value):
    if isinstance(value, list):
        return set(value)
    else:
        return value

def detect_changes_by_format(old_data: Dict[str, Any], new_data: Dict[str, Any], format_str: str) -> bool:
    output_keys = [match.group(1)[:-7] for match in re.finditer(r'{(\w+_output)}', format_str)]

    for key in output_keys:
        old_value = normalize_value(old_data.get(key))
        new_value = normalize_value(new_data.get(key))

        if old_value != new_value:
            return True

    return False

def clean_output(output: str) -> str:
    output = re.sub(r'\s*\|\s*', '|', output)
    output = re.sub(r'\|{2,}', '|', output)
    output = re.sub(r'\s*@@\s*', '@@', output)
    output = output.strip()
    return output