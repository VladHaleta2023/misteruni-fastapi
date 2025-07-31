from g4f.client import Client
from typing import Optional, Dict, Any
import re
import copy

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

data = {
    'changed': 'true',
    'subject': 'Matematyka',
    'section': 'Liczby rzeczywiste',
    'topic': 'Liczby naturalne',
    'subtopics': [],
    'attempt': 0,
    'prompt': """
Jesteś ekspertem edukacyjnym specjalizującym się w tworzeniu kompletnej i logicznie uporządkowanej siatki tematycznej. Twoim zadaniem jest wygenerowanie **szczegółowych, kompletnych i jednoznacznie zdefiniowanych podtematów** dla:

- przedmiotu: "{subject_input}",  
- rozdziału: "{section_input}",  
- tematu: "{topic_input}".

Aktualne podtematy to: {subtopics_input}

---

### Wymagania dotyczące podtematów:

1. **Zgodność z podstawą programową MEN**:  
   - Podtematy muszą być w pełni zgodne z **obowiązującą podstawą programową Ministerstwa Edukacji Narodowej** dla **szkoły średniej (liceum i technikum)**.  
   - Zagadnienia muszą odpowiadać poziomowi nauczania zgodnie z programem przedmiotu.

2. **Pełne pokrycie materiału**:  
   - Lista musi obejmować **100% zagadnień** wynikających z tematu i wymaganych przez MEN.  
   - Upewnij się, że podtematy pokrywają wszystkie aspekty tematu wymienione w podstawie programowej MEN oraz nie pomijają żadnego kluczowego zagadnienia.

3. **Precyzyjna i konkretna struktura**:  
   - Każdy podtemat to **konkretne, odrębne zagadnienie**, które można niezależnie omawiać.  
   - Unikaj zbyt ogólnych, wieloznacznych lub powtarzających się sformułowań.  
   - Nie twórz tematów zamkniętych logicznie w sobie lub o zbliżonym znaczeniu (np. „Rodzaje równań i ich podział” oraz „Podział równań ze względu na rodzaje” nie mogą występować razem).

4. **Brak tautologii, powtórzeń i synonimów**:  
   - Wyklucz tematy powtarzające się znaczeniowo lub stylistycznie.  
   - Nie używaj podtematów o zbliżonym znaczeniu lub powtarzających się treściach.

5. **Spójność logiczna i tematyczna**:  
   - Wszystkie podtematy muszą być **spójne logicznie, merytorycznie i tematycznie** z tematami `{topic_input}`, `{section_input}` i `{subject_input}`.  
   - Nie mogą się powtarzać ani nawiązywać do innych tematów czy działów.

6. **Wzory matematyczne w LaTeX (jeśli wymagane)**:  
   - Używaj poprawnej składni LaTeX (`$...$` inline, `$$...$$` block) przy wszystkich wyrażeniach matematycznych.  
   - Wzory muszą być zgodne z poziomem szkoły średniej i gotowe do parsowania przez KaTeX.  
   - Stosuj wzory tylko wtedy, gdy są one niezbędne do precyzyjnego zdefiniowania podtematu.

7. **Formatowanie podtematów**:  
   - Każdy podtemat musi składać się z poprawnie zapisanych wyrazów, oddzielonych pojedynczymi spacjami (nie zlepione razem).  
   - Podtematy muszą być oddzielone wyłącznie znakiem `|`, bez spacji przed i po tym znaku.  
   - Przykład poprawnego formatu: `Podtemat pierwszy|Podtemat drugi|Podtemat trzeci`.

8. **Styl odpowiedzi**:  
   - Nie dodawaj żadnych nagłówków, opisów, komentarzy ani nowych linii.  
   - Zwróć **jedynie ciąg znaków nazw podtematów oddzielonych znakiem `|`**, bez nawiasów `{}`, dodatkowych spacji, cudzysłowów czy innych znaków specjalnych.

9. **Unikalność i sortowanie**:  
   - Podtematy muszą być unikalne (bez duplikatów).  
   - Podtematy powinny być posortowane alfabetycznie.

10. **Kontrola jakości**:  
    - Przed wygenerowaniem listy zweryfikuj, czy:  
      a) lista jest kompletna i pokrywa wszystkie wymagane aspekty,  
      b) podtematy są jednoznaczne i unikalne,  
      c) nie ma powtórzeń ani tematów wykraczających poza zakres `{topic_input}`.

---

### Format odpowiedzi (wygeneruj tylko tę linię jako output):  
Odpowiedz wyłącznie jako ciąg znaków nazw podtematów, oddzielonych znakiem `|`, bez żadnych dodatkowych spacji, cudzysłowów, nawiasów, nagłówków czy komentarzy.

{subtopics_output}"""
}

MAX_ATTEMPTS = 10

while data['changed'].lower() == 'true' and data['attempt'] < MAX_ATTEMPTS:
    old_data = copy.deepcopy(data)
    prompt = metadata_fill_template(data['prompt'], data)
    response = request_ai(prompt)

    if not response:
        print("AI Response is None")
        break
    else:
        response = clean_output(response)

    format_str = extract_format_from_prompt(prompt)
    data = parse_output_by_format(format_str, response, data)
    data['attempt'] += 1

    print(old_data['subtopics'])
    print(data['subtopics'])

    if detect_changes_by_format(old_data, data, format_str):
        data['changed'] = 'true'
    else:
        data['changed'] = 'false'

print("Final result:", data['subtopics'])