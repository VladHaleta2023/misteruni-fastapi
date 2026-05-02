import logging
import re
from typing import List
from enum import Enum

class SubjectDetailLevel(str, Enum):
    BASIC = "BASIC"
    EXPANDED = "EXPANDED"
    ACADEMIC = "ACADEMIC"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FORBIDDEN_ENVS = ['align', 'equation', 'array', 'matrix', 'multline', 'gather', 'flalign']

def remove_duplicates(subtopics: List[str]) -> List[str]:
    seen = set()
    unique = []
    for line in subtopics:
        normalized = line.strip()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique

def validate_latex(line: str, errors: list) -> bool:
    patterns = [
        r'\$\$(.*?)\$\$',
        r'\\\[(.*?)\\\]',
        r'\\\((.*?)\\\)',
        r'\$(.*?)\$'
    ]

    for pat in patterns:
        for match in re.finditer(pat, line, re.DOTALL):
            formula_content = match.group(1)
            for env in FORBIDDEN_ENVS:
                if re.search(r'\\begin\{' + env + r'\}', formula_content) or \
                   re.search(r'\\end\{' + env + r'\}', formula_content):
                    errors.append(f"Niedozwolone środowisko LaTeX '{env}' w formule: {match.group(0)}")
                    return False

    temp_line = re.sub(r'\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|\$.*?\$', '', line, flags=re.DOTALL)
    for env in FORBIDDEN_ENVS:
        if re.search(r'\\begin\{' + env + r'\}', temp_line) or \
           re.search(r'\\end\{' + env + r'\}', temp_line):
            errors.append(f"Niedozwolone środowisko LaTeX '{env}' poza formułą w linii: {line}")
            return False

    return True

def remove_empty_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line.strip()]

def find_last_semicolon_outside_braces(s: str) -> int:
    depth = 0
    for i in reversed(range(len(s))):
        if s[i] == '}':
            depth += 1
        elif s[i] == '{':
            depth -= 1
        elif s[i] == ';' and depth == 0:
            return i
    return -1

def extract_wrong_words(words: list) -> list:
    wrong_words = []
    for item in words:
        if ";" in item:
            wrong_words.append(item.split(";")[0].strip())
        else:
            wrong_words.append(item.strip())
    return wrong_words

def parse_subtopics_response(
    old_subtopics: list,
    response: str,
    errors: list,
    percent_message: str="Ocena ważności",
    blockStart: str="Start:",
    blockEnd: str = "End:"
) -> list:
    try:
        start_idx = response.find(f"{blockStart}")
        end_idx = response.find(f"{blockEnd}", start_idx)
        if start_idx == -1:
            errors.append(f"Błąd parsowania: brak etykiety {blockStart}")
            return old_subtopics
        if end_idx == -1:
            errors.append(f"Błąd parsowania: brak etykiety {blockEnd}")
            return old_subtopics
        if end_idx <= start_idx:
            errors.append(f"Błąd parsowania: etykieta {blockEnd} znajduje się przed {blockStart}")
            return old_subtopics

        content = response[start_idx + len(blockStart): end_idx].strip()
        if not content:
            errors.append(f"Błąd parsowania: brak podtematów pomiędzy {blockStart} a {blockEnd}")
            return old_subtopics

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        lines = remove_empty_lines(lines)
        unique_lines = list(dict.fromkeys(lines))
        if len(unique_lines) < len(lines):
            errors.append("Usunięto powtarzające się podtematy.")

        final_subtopics = []
        has_error = False

        for line in unique_lines:
            if re.search(r"\s;\s|\s;|;\s", line):
                errors.append(f"Błąd formatu podtematu (spacje wokół ';' są niedozwolone): '{line}'")
                has_error = True
                continue
            semicolon_idx = find_last_semicolon_outside_braces(line)
            if semicolon_idx == -1:
                errors.append(f"Błąd formatu podtematu (brak znaku ';'): '{line}'")
                continue

            name = line[:semicolon_idx]
            score_str = line[semicolon_idx + 1:]

            if name != name.strip():
                errors.append(f"Nazwa podtematu zawiera białe znaki na początku lub końcu: '{name}'")
                has_error = True
                continue
            if not validate_latex(name, errors):
                errors.append(f"Błąd LaTeX w podtemacie: '{line}'")
                has_error = True
                continue

            score_str = score_str.replace('%', '').strip()

            try:
                if score_str == "":
                    errors.append(f"{percent_message} jest pusta w podtemacie '{line}'")
                    has_error = True
                    continue

                score = int(score_str)
            except ValueError:
                errors.append(f"{percent_message} nie jest liczbą całkowitą: '{score_str}' w podtemacie '{line}'")
                has_error = True
                continue

            final_subtopics.append([name, score])

        if has_error and not final_subtopics:
            errors.append("Wszystkie podtematy zostały odrzucone ze względu na błędy formatowania.")
            return old_subtopics

        return final_subtopics

    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania podtematów: {str(e)}")
        return old_subtopics

def parse_subtopics_status_response(old_subtopics: list, response: str, errors: list) -> list:
    try:
        start_idx = response.find("Start:")
        end_idx = response.find("End:", start_idx)
        if start_idx == -1:
            errors.append("Błąd parsowania: brak etykiety Start:")
            return old_subtopics
        if end_idx == -1:
            errors.append("Błąd parsowania: brak etykiety End:")
            return old_subtopics
        if end_idx <= start_idx:
            errors.append("Błąd parsowania: etykieta End: znajduje się przed Start:")
            return old_subtopics

        content = response[start_idx + len("Start:"): end_idx].strip()
        if not content:
            errors.append("Błąd parsowania: brak podtematów pomiędzy Start: a End:")
            return old_subtopics

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        lines = remove_empty_lines(lines)
        unique_lines = list(dict.fromkeys(lines))
        if len(unique_lines) < len(lines):
            errors.append("Usunięto powtarzające się podtematy.")

        final_subtopics = []
        has_error = False

        for line in unique_lines:
            if re.search(r"\s;\s|\s;|;\s", line):
                errors.append(f"Błąd formatu podtematu (spacje wokół ';' są niedozwolone): '{line}'")
                has_error = True
                continue
            semicolon_idx = find_last_semicolon_outside_braces(line)
            if semicolon_idx == -1:
                errors.append(f"Błąd formatu podtematu (brak znaku ';'): '{line}'")
                continue

            name = line[:semicolon_idx]
            status = line[semicolon_idx + 1:]

            if name != name.strip():
                errors.append(f"Nazwa podtematu zawiera białe znaki na początku lub końcu: '{name}'")
                has_error = True
                continue
            if not validate_latex(name, errors):
                errors.append(f"Błąd LaTeX w podtemacie: '{line}'")
                has_error = True
                continue
            if status not in SubjectDetailLevel.__members__:
                errors.append(f"Nieprawidłowy status: {status}")
                has_error = True
                continue
            if status == "":
                errors.append(f"Status jest pusty w podtemacie '{line}'")
                has_error = True
                continue

            final_subtopics.append([name, status])

        if has_error and not final_subtopics:
            errors.append("Wszystkie podtematy zostały odrzucone ze względu na błędy formatowania.")
            return old_subtopics

        return final_subtopics

    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania podtematów: {str(e)}")
        return old_subtopics

def parse_words_response(old_words: list, response: str, errors: list, percent_message: str="Częstotliwość słowa tematycznego") -> list:
    try:
        start_idx = response.find("Start:")
        end_idx = response.find("End:", start_idx)
        if start_idx == -1:
            errors.append("Błąd parsowania: brak etykiety Start:")
            return old_words
        if end_idx == -1:
            errors.append("Błąd parsowania: brak etykiety End:")
            return old_words
        if end_idx <= start_idx:
            errors.append("Błąd parsowania: etykieta End: znajduje się przed Start:")
            return old_words

        content = response[start_idx + len("Start:"): end_idx].strip()
        if not content:
            errors.append("Błąd parsowania: brak słów tematycznych pomiędzy Start: a End:")
            return old_words

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        lines = remove_empty_lines(lines)
        unique_lines = list(dict.fromkeys(lines))
        if len(unique_lines) < len(lines):
            errors.append("Usunięto powtarzające się słowy tematyczne.")

        final_words = []
        has_error = False

        for line in unique_lines:
            if re.search(r"\s;\s|\s;|;\s", line):
                errors.append(f"Błąd formatu słowa tematycznego (spacje wokół ';' są niedozwolone): '{line}'")
                has_error = True
                continue
            semicolon_idx = find_last_semicolon_outside_braces(line)
            if semicolon_idx == -1:
                errors.append(f"Błąd formatu słowa tematycznego (brak znaku ';'): '{line}'")
                continue

            name = line[:semicolon_idx]
            score_str = line[semicolon_idx + 1:]

            if "%" in score_str:
                errors.append(f"{percent_message} nie może zawierać '%': '{score_str}' w słowie tematycznym '{line}'")
                has_error = True
                continue

            try:
                if score_str == "":
                    errors.append(f"{percent_message} jest pusta w słowie tematycznym '{line}'")
                    has_error = True
                    continue

                score = int(score_str)
            except ValueError:
                errors.append(f"{percent_message} nie jest liczbą całkowitą: '{score_str}' w słowie tematycznym '{line}'")
                has_error = True
                continue

            final_words.append([name, score])

        if has_error and not final_words:
            errors.append("Wszystkie słowy tematyczne zostały odrzucone ze względu na błędy formatowania.")
            return old_words

        return final_words

    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania podtematów: {str(e)}")
        return old_words

def parse_task_response(old_text: str, response: str, errors: list) -> str:
    try:
        response = response.replace('\r\n', '\n').strip()

        start_match = re.search(r'Start\s*:', response, re.IGNORECASE)
        end_match = re.search(r'End\s*:', response, re.IGNORECASE)

        if not start_match:
            errors.append("Błąd parsowania: brak etykiety Start:")
            return old_text
        if not end_match:
            errors.append("Błąd parsowania: brak etykiety End:")
            return old_text
        if end_match.start() <= start_match.end():
            errors.append("Błąd parsowania: etykieta End: znajduje się przed Start:")
            return old_text

        final_text = response[start_match.end(): end_match.start()].strip()

        if not final_text:
            errors.append("Błąd: tekst zadania jest pusty")
            return old_text

        if not validate_latex(final_text, errors):
            errors.append(f"Błąd LaTeX w tekście zadania: '{final_text}'")
            return old_text

        return final_text
    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania: {str(e)}")
        return old_text

def parse_translate_response(old_translate: str, response: str, errors: list) -> str:
    try:
        response = response.replace('\r\n', '\n').strip()

        start_match = re.search(r'Start\s*:', response, re.IGNORECASE)
        end_match = re.search(r'End\s*:', response, re.IGNORECASE)

        if not start_match:
            errors.append("Błąd parsowania: brak etykiety Start:")
            return old_translate
        if not end_match:
            errors.append("Błąd parsowania: brak etykiety End:")
            return old_translate
        if end_match.start() <= start_match.end():
            errors.append("Błąd parsowania: etykieta End: znajduje się przed Start:")
            return old_translate

        final_translate = response[start_match.end(): end_match.start()].strip()

        if not final_translate:
            errors.append("Błąd: tekst zadania jest pusty")
            return old_translate

        return final_translate
    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania: {str(e)}")
        return old_translate

def parse_chat_response(old_chat: str, response: str, errors: list) -> str:
    try:
        response = response.replace('\r\n', '\n').strip()

        start_match = re.search(r'<chat>', response, re.IGNORECASE)
        end_match = re.search(r'</chat>', response, re.IGNORECASE)

        if not start_match:
            errors.append("Błąd parsowania: brak etykiety <chat>")
            return old_chat
        if not end_match:
            errors.append("Błąd parsowania: brak etykiety </chat>")
            return old_chat
        if end_match.start() <= start_match.end():
            errors.append("Błąd parsowania: etykieta </chat> znajduje się przed <chat>")
            return old_chat

        final_chat = response[start_match.end(): end_match.start()].strip()

        if not final_chat:
            errors.append("Błąd: tekst czatu jest pusty")
            return old_chat

        return final_chat
    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania: {str(e)}")
        return old_chat

def parse_literature_response(old_note: str, response: str, errors: list) -> str:
    try:
        response = response.replace('\r\n', '\n').strip()

        start_match = re.search(r'<literature>', response, re.IGNORECASE)
        end_match = re.search(r'</literature>', response, re.IGNORECASE)

        if not start_match:
            errors.append("Błąd parsowania: brak etykiety <literature>")
            return old_note
        if not end_match:
            errors.append("Błąd parsowania: brak etykiety </literature>")
            return old_note
        if end_match.start() <= start_match.end():
            errors.append("Błąd parsowania: etykieta </literature> znajduje się przed <literature>")
            return old_note

        final_chat = response[start_match.end(): end_match.start()].strip()

        if not final_chat:
            errors.append("Błąd: tekst literatury jest pusty")
            return old_note

        return final_chat
    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania: {str(e)}")
        return old_note

def parse_note_response(old_note: str, response: str, errors: list) -> str:
    try:
        response = response.replace('\r\n', '\n').strip()

        start_match = re.search(r'noteStart\s*:', response, re.IGNORECASE)
        end_match = re.search(r'noteEnd\s*:', response, re.IGNORECASE)

        if not start_match:
            errors.append("Błąd parsowania: brak etykiety noteStart:")
            return old_note
        if not end_match:
            errors.append("Błąd parsowania: brak etykiety noteEnd:")
            return old_note
        if end_match.start() <= start_match.end():
            errors.append("Błąd parsowania: etykieta noteEnd: znajduje się przed noteStart:")
            return old_note

        final_text = response[start_match.end(): end_match.start()].strip()

        if not final_text:
            errors.append("Błąd: notatka zadania jest pusta")
            return old_note

        if not validate_latex(final_text, errors):
            errors.append(f"Błąd LaTeX w notatce zadania: '{final_text}'")
            return old_note
        return final_text
    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania: {str(e)}")
        return old_note

def parse_solution_guide_response(old_solution_guide: str, response: str, errors: list) -> str:
    try:
        response = response.replace('\r\n', '\n').strip()

        start_match = re.search(r'Start\s*:', response, re.IGNORECASE)
        end_match = re.search(r'End\s*:', response, re.IGNORECASE)

        if not start_match:
            errors.append("Błąd parsowania: brak etykiety Start:")
            return old_solution_guide
        if not end_match:
            errors.append("Błąd parsowania: brak etykiety End:")
            return old_solution_guide
        if end_match.start() <= start_match.end():
            errors.append("Błąd parsowania: etykieta End: znajduje się przed Start:")
            return old_solution_guide

        final_text = response[start_match.end(): end_match.start()].strip()

        if not final_text:
            errors.append("Błąd: przewodnik rozwiązania zadania jest pusty")
            return old_solution_guide

        if not validate_latex(final_text, errors):
            errors.append(f"Błąd LaTeX w przewodniku rozwiązania zadania: '{final_text}'")
            return old_solution_guide
        return final_text
    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania: {str(e)}")
        return old_solution_guide

def parse_frequency_response(old_frequency: int, response: str, errors: list) -> int:
    try:
        response = response.replace('\r\n', '\n').strip()

        start_match = re.search(r'frequencyStart\s*:', response, re.IGNORECASE)
        end_match = re.search(r'frequencyEnd\s*:', response, re.IGNORECASE)

        if not start_match:
            errors.append("Błąd parsowania: brak etykiety frequencyStart:")
            return old_frequency
        if not end_match:
            errors.append("Błąd parsowania: brak etykiety frequencyEnd:")
            return old_frequency
        if end_match.start() <= start_match.end():
            errors.append("Błąd parsowania: etykieta frequencyEnd: znajduje się przed frequencyStart:")
            return old_frequency

        freq_text = response[start_match.end(): end_match.start()].strip()

        if not freq_text:
            errors.append("Błąd: blok częstotliwości jest pusty")
            return old_frequency

        if not re.fullmatch(r'\d{1,3}', freq_text):
            errors.append(f"Błąd: niepoprawny format liczby w frequency: '{freq_text}'")
            return old_frequency

        freq_value = int(freq_text)
        if not (0 <= freq_value <= 100):
            errors.append(f"Błąd: liczba frequency {freq_value} poza zakresem 0–100")
            return old_frequency

        return freq_value
    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania frequency: {str(e)}")
        return old_frequency

def parse_explanation_response(old_explanation: str, response: str, errors: list,
                               output_subtopics: list, correctOption: str, userOption: str,
                               topic: str, type: str) -> str:
    try:
        response = response.replace('\r\n', '\n').strip()

        start_match = re.search(r'explanationStart\s*:', response, re.IGNORECASE)
        end_match = re.search(r'explanationEnd\s*:', response, re.IGNORECASE)

        if not start_match:
            errors.append("Błąd parsowania: brak etykiety explanationStart:")
            return old_explanation
        if not end_match:
            errors.append("Błąd parsowania: brak etykiety explanationEnd:")
            return old_explanation
        if end_match.start() <= start_match.end():
            errors.append("Błąd parsowania: etykieta explanationEnd znajduje się przed explanationStart:")
            return old_explanation

        final_text = response[start_match.end(): end_match.start()].strip()
        if not final_text:
            return old_explanation

        pattern = r"(\*\*.+?:\*\*)\n❓ (.+?)(?=\n\*\*|$)"
        matches = re.findall(pattern, final_text, flags=re.DOTALL)

        if not matches:
            return old_explanation

        work_on_label = "❓"
        if type == "Stories":
            task_score_label = "Opowiadanie"
            subtopic_score_label = "Opowiadanie"
        else:
            task_score_label = "Ocena:"
            subtopic_score_label = "Ocena:"

        is_single_topic_match = (
                len(output_subtopics) == 1 and
                len(output_subtopics[0]) >= 1 and
                output_subtopics[0][0] == topic
        )

        new_final_text = ""
        for i, match in enumerate(matches):
            if len(match) == 2:
                topic_name_in_match, explanation = match
            else:
                topic_name_in_match = match[0] if match else ""
                explanation = final_text

            topic_name_clean = topic_name_in_match.strip('*: ')
            percent_error = 0

            if output_subtopics:
                for subtopic_name, error in output_subtopics:
                    if subtopic_name == topic_name_clean:
                        percent_error = float(error)
                        break

            bonus = 20 if correctOption == userOption else 0
            new_percent = round((100 - percent_error) * 0.8 + bonus)

            if is_single_topic_match and topic_name_clean == topic:
                score_label = task_score_label
            else:
                score_label = subtopic_score_label

            new_final_text += f"{topic_name_in_match}\n{work_on_label} {explanation.strip()}\n\n{score_label} {new_percent}%\n\n"

        return new_final_text.strip()
    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania: {str(e)}")
        return old_explanation

def parse_writing_explanation(old_explanation: str, response: str, errors: list) -> str:
    try:
        response = response.replace('\r\n', '\n').strip()

        start_match = re.search(r'explanationStart\s*:', response, re.IGNORECASE)
        end_match = re.search(r'explanationEnd\s*:', response, re.IGNORECASE)

        if not start_match:
            errors.append("Błąd parsowania: brak etykiety explanationStart:")
            return old_explanation
        if not end_match:
            errors.append("Błąd parsowania: brak etykiety explanationEnd:")
            return old_explanation
        if end_match.start() <= start_match.end():
            errors.append("Błąd parsowania: etykieta explanationEnd znajduje się przed explanationStart:")
            return old_explanation

        final_text = response[start_match.end(): end_match.start()].strip()

        if not final_text:
            return old_explanation

        return final_text

    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania: {str(e)}")
        return old_explanation

def parse_words_output_text_response(old_text: str, response: str, errors: list) -> str:
    try:
        response = response.replace('\r\n', '\n').strip()

        start_match = re.search(r'Start\s*:', response, re.IGNORECASE)
        end_match = re.search(r'End\s*:', response, re.IGNORECASE)

        if not start_match:
            errors.append("Błąd parsowania: brak etykiety Start:")
            return old_text
        if not end_match:
            errors.append("Błąd parsowania: brak etykiety End:")
            return old_text
        if end_match.start() <= start_match.end():
            errors.append("Błąd parsowania: etykieta End: znajduje się przed Start:")
            return old_text

        final_text = response[start_match.end(): end_match.start()].strip()

        if not final_text:
            errors.append("Błąd: tekst pojaśnienia jest pusty")
            return old_text

        return final_text

    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania: {str(e)}")
        return old_text

def parse_interactive_task_text_response(old_text: str, response: str, errors: list) -> str:
    try:
        response = response.replace('\r\n', '\n').strip()

        start_match = re.search(r'Start\s*:', response, re.IGNORECASE)
        end_match = re.search(r'End\s*:', response, re.IGNORECASE)

        if not start_match:
            errors.append("Błąd parsowania: brak etykiety Start:")
            return old_text
        if not end_match:
            errors.append("Błąd parsowania: brak etykiety End:")
            return old_text
        if end_match.start() <= start_match.end():
            errors.append("Błąd parsowania: etykieta End: znajduje się przed Start:")
            return old_text

        final_text = response[start_match.end(): end_match.start()].strip()

        if not final_text:
            errors.append("Błąd: tekst opowiadania jest pusty")
            return old_text

        return final_text

    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania: {str(e)}")
        return old_text

def parse_interactive_task_translate_response(old_translate: str, response: str, errors: list) -> str:
    try:
        response = response.replace('\r\n', '\n').strip()

        start_match = re.search(r'translateStart\s*:', response, re.IGNORECASE)
        end_match = re.search(r'translateEnd\s*:', response, re.IGNORECASE)

        if not start_match:
            errors.append("Błąd parsowania: brak etykiety translateStart:")
            return old_translate
        if not end_match:
            errors.append("Błąd parsowania: brak etykiety translateEnd:")
            return old_translate
        if end_match.start() <= start_match.end():
            errors.append("Błąd parsowania: etykieta translateEnd: znajduje się przed translateStart:")
            return old_translate

        final_translate = response[start_match.end(): end_match.start()].strip()

        if not final_translate:
            errors.append("Błąd: tekst tłumaczenia opowiadania jest pusty")
            return old_translate
        return final_translate

    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania: {str(e)}")
        return old_translate

def parse_options_response(old_data: dict, response: str, errors: list) -> dict:
    final_data = {
        "options": old_data.get("options", [])
    }

    try:
        start_idx = response.find("Start:")
        end_idx = response.find("End:", start_idx)
        if start_idx == -1:
            errors.append("Błąd parsowania: brak etykiety Start:")
            return final_data
        if end_idx == -1:
            errors.append("Błąd parsowania: brak etykiety End:")
            return final_data
        if end_idx <= start_idx:
            errors.append("Błąd parsowania: etykieta End: znajduje się przed Start:")
            return final_data

        content = response[start_idx + len("Start:"): end_idx].strip()
        if not content:
            errors.append("Błąd parsowania: brak wariantów oraz numeru lub index prawidłowej odpowiedzi pomiędzy Start: a End:")
            return final_data

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        unique_lines = list(dict.fromkeys(lines))
        if len(unique_lines) < len(lines):
            errors.append("Usunięto powtarzające się warianty.")

        if len(unique_lines) < 4:
            errors.append(f"Za mało linii: {len(unique_lines)}, powinno być co najmniej 4 warianty")
            return final_data

        final_data['options'] = []
        for line in unique_lines:
            if validate_latex(line, errors):
                final_data['options'].append(line)
            else:
                errors.append(f"Błąd LaTeX wariantu: '{line}'")

        if final_data['options']:
            first_len = len(final_data['options'][0])
            other_lens = [len(opt) for opt in final_data['options'][1:]]
            if any(first_len > l for l in other_lens):
                errors.append("Pierwszy prawidłowy wariant jest dłuższy niż pozostałe warianty, co może wskazywać na problem.")

        return final_data

    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania: {str(e)}")
        return final_data

def parse_output_subtopics_response_filtered(old_subtopics: list, new_subtopics: list, subtopics: list, errors: list) -> list:
    filtered_subtopics = []

    for name, score in new_subtopics:
        if name not in subtopics:
            errors.append(f"Podtemat '{name}' nie znajduje się w liście subtopics.")
        else:
            filtered_subtopics.append([name, score])

    if not filtered_subtopics:
        return old_subtopics

    return filtered_subtopics

def parse_output_subtopics_response(old_subtopics: list, subtopics: list, response: str, errors: list) -> list:
    try:
        start_idx = response.find("subtopicsStart:")
        end_idx = response.find("subtopicsEnd:", start_idx)
        if start_idx == -1:
            errors.append("Błąd parsowania: brak etykiety subtopicsStart:")
            return old_subtopics
        if end_idx == -1:
            errors.append("Błąd parsowania: brak etykiety subtopicsEnd:")
            return old_subtopics

        content = response[start_idx + len("subtopicsStart:"): end_idx].strip()
        if not content:
            errors.append("Błąd parsowania: brak podtematów pomiędzy subtopicsStart: a subtopicsEnd:")
            return old_subtopics

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        extracted_names = []

        for line in lines:
            parts = line.split(";")
            name = parts[0].strip()

            if name:
                extracted_names.append(name)

        if not extracted_names:
            errors.append("Brak podtematów do przetworzenia.")
            return old_subtopics

        unique_names = []
        seen = set()
        for name in extracted_names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)

        subtopics_names = [s[0] if isinstance(s, (list, tuple)) else s.split(";")[0].strip() for s in subtopics]
        final_subtopics = []

        for name in unique_names:
            if not validate_latex(name, errors):
                errors.append(f"Błąd LaTeX w podtemacie: '{name}'")
                continue

            if name in subtopics_names:
                final_subtopics.append(name)
            else:
                errors.append(f"Podtemat '{name}' nie znajduje się w liście subtopics.")

        return final_subtopics if final_subtopics else old_subtopics

    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania podtematów: {str(e)}")
        return old_subtopics

def parse_output_words_response(
    old_words: list,
    words: list,
    response: str,
    errors: list,
    words_are_tuples: bool = True
) -> list:
    try:
        start_idx = response.find("<words>")
        end_idx = response.find("</words>", start_idx)
        if start_idx == -1:
            errors.append("Błąd parsowania: brak etykiety <words>")
            return old_words
        if end_idx == -1:
            errors.append("Błąd parsowania: brak etykiety </words>")
            return old_words
        if end_idx <= start_idx:
            errors.append("Błąd parsowania: etykieta </words> znajduje się przed <words>")
            return old_words

        content = response[start_idx + len("<words>"): end_idx].strip()
        if not content:
            errors.append("Błąd parsowania: brak wyrazów pomiędzy <words> a </words>")
            return old_words

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        lines = extract_wrong_words(lines)
        lines = remove_empty_lines(lines)
        unique_lines = list(dict.fromkeys(lines))

        filtered_words = []

        for name in unique_lines:
            name_lower = name.lower()
            if words_are_tuples:
                if any(name_lower == w[0].lower() for w in words):
                    filtered_words.append(name)
            else:
                if any(name_lower == w.lower() for w in words):
                    filtered_words.append(name)

        if not filtered_words:
            return old_words

        return filtered_words

    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania wyrazów: {str(e)}")
        return old_words

def remove_markdown(text: str) -> str:
    if not text:
        return text

    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)

    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)

    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)

    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)

    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)

    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()