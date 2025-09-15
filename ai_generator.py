import anti_gui

import logging
import re
from typing import List

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

def parse_subtopics_response(old_subtopics: list, response: str, errors: list, percent_message: str="Ocena ważności") -> list:
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
            score_str = line[semicolon_idx + 1:]

            if name != name.strip():
                errors.append(f"Nazwa podtematu zawiera białe znaki na początku lub końcu: '{name}'")
                has_error = True
                continue
            if not validate_latex(name, errors):
                errors.append(f"Błąd LaTeX w podtemacie: '{line}'")
                has_error = True
                continue
            if "%" in score_str:
                errors.append(f"{percent_message} nie może zawierać '%': '{score_str}' w podtemacie '{line}'")
                has_error = True
                continue

            try:
                if score_str == "":
                    errors.append(f"{percent_message} jest pusta w podtemacie '{line}'")
                    has_error = True
                    continue

                score = int(score_str)
                if not (0 <= score <= 100):
                    errors.append(f"{percent_message} poza zakresem 0-100: '{score_str}' w podtemacie '{line}'")
                    has_error = True
                    continue
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

def parse_questions_response(old_questions: list, response: str, errors: list) -> list:
    try:
        start_idx = response.find("Start:")
        end_idx = response.find("End:", start_idx)
        if start_idx == -1:
            errors.append("Błąd parsowania: brak etykiety Start:")
            return old_questions
        if end_idx == -1:
            errors.append("Błąd parsowania: brak etykiety End:")
            return old_questions
        if end_idx <= start_idx:
            errors.append("Błąd parsowania: etykieta End: znajduje się przed Start:")
            return old_questions

        content = response[start_idx + len("Start:"): end_idx].strip()
        if not content:
            errors.append("Błąd parsowania: brak pytań pomiędzy Start: a End:")
            return old_questions

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        lines = remove_empty_lines(lines)
        unique_lines = list(dict.fromkeys(lines))
        if len(unique_lines) < len(lines):
            errors.append("Usunięto powtarzające się pytania.")

        return unique_lines
    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania pytań: {str(e)}")
        return old_questions

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

        if any(c in final_text for c in ['\n', '\r']):
            errors.append("Błąd: tekst zadania zawiera nowe linie ('\\n'), niezgodne z wymaganiami")
            return old_text

        if not validate_latex(final_text, errors):
            errors.append(f"Błąd LaTeX w tekście zadania: '{final_text}'")
            return old_text

        def remove_latex_segments(text):
            text = re.sub(r'\$.*?\$', '', text)
            text = re.sub(r'\\\[.*?\\\]', '', text)
            return text

        text_no_latex = remove_latex_segments(final_text)
        pattern = r'\b([A-Da-d1-4ivxIVX])[\)\.\-:]'
        if re.search(pattern, text_no_latex):
            errors.append("Błąd: tekst zadania zawiera potencjalne warianty odpowiedzi (np. A), B), 1.), i), etc.)")
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

def parse_solution_response(old_solution: str, response: str, errors: list) -> str:
    try:
        start_idx = response.find("Start:")
        end_idx = response.find("End:", start_idx)

        if start_idx == -1:
            errors.append("Błąd parsowania: brak etykiety Start:")
            return old_solution
        if end_idx == -1:
            errors.append("Błąd parsowania: brak etykiety End:")
            return old_solution
        if end_idx <= start_idx:
            errors.append("Błąd parsowania: etykieta End: znajduje się przed Start:")
            return old_solution

        final_solution = response[start_idx + len("Start:"): end_idx].strip()

        if not final_solution:
            errors.append("Błąd: rozwiązanie zadania jest puste")
            return old_solution

        if '\n' in final_solution:
            errors.append("Błąd: rozwiązanie zadania zawiera nowe linie ('\\n'), niezgodne z wymaganiami")
            return old_solution

        if not validate_latex(final_solution, errors):
            errors.append(f"Błąd LaTeX w rozwiązaniu zadania: '{final_solution}'")
            return old_solution

        return final_solution

    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania: {str(e)}")
        return old_solution

def parse_options_response(old_data: dict, response: str, errors: list) -> dict:
    final_data = {
        "options": old_data.get("options", []),
        "correctOptionIndex": old_data.get("correctOptionIndex", 0)
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
        lines = remove_empty_lines(lines)
        unique_lines = list(dict.fromkeys(lines))
        if len(unique_lines) < len(lines):
            errors.append("Usunięto powtarzające się warianty.")

        if len(unique_lines) < 2:
            errors.append(f"Za mało linii: {len(unique_lines)}, powinno być co najmniej 2 (min. 1 wariant + 1 numer).")
            return final_data

        options_lines = unique_lines[:-1]
        score_line = unique_lines[-1]

        final_data['options'] = []
        for line in options_lines:
            if validate_latex(line, errors):
                final_data['options'].append(line)
            else:
                errors.append(f"Błąd LaTeX wariantu: '{line}'")

        if len(final_data['options']) != 4:
            errors.append(f"Niepoprawna liczba wariantów ({len(final_data['options'])}), oczekiwano 4.")

        try:
            score = int(score_line)
            if 0 <= score < len(final_data['options']):
                final_data['correctOptionIndex'] = score
            else:
                errors.append(f"Numer prawidłowej odpowiedzi poza zakresem 0-{len(final_data['options'])-1}: '{score_line}'")
        except ValueError:
            errors.append(f"Numer lub index prawidłowej odpowiedzi nie jest liczbą całkowitą: '{score_line}'")

        return final_data

    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania podtematów: {str(e)}")
        return final_data

def parse_output_subtopics_response(old_subtopics: list, new_subtopics: list, subtopics: list, errors: list) -> list:
    filtered_subtopics = []

    for name, score in new_subtopics:
        if name not in subtopics:
            errors.append(f"Podtemat '{name}' nie znajduje się w liście subtopics.")
        else:
            filtered_subtopics.append([name, score])

    if not filtered_subtopics:
        return old_subtopics

    return filtered_subtopics

def parse_task_output_subtopics_response(old_subtopics: list, subtopics: list, response: str, errors: list) -> list:
    try:
        start_idx = response.find("subtopicsStart:")
        end_idx = response.find("subtopicsEnd:", start_idx)
        if start_idx == -1:
            errors.append("Błąd parsowania: brak etykiety subtopicsStart:")
            return old_subtopics
        if end_idx == -1:
            errors.append("Błąd parsowania: brak etykiety subtopicsEnd:")
            return old_subtopics
        if end_idx <= start_idx:
            errors.append("Błąd parsowania: etykieta subtopicsEnd: znajduje się przed subtopicsStart:")
            return old_subtopics

        content = response[start_idx + len("subtopicsStart:"): end_idx].strip()
        if not content:
            errors.append("Błąd parsowania: brak podtematów pomiędzy subtopicsStart: a subtopicsEnd:")
            return old_subtopics

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        lines = remove_empty_lines(lines)
        filtered_lines = []

        for line in lines:
            clean_line = line.split(";")[0].strip()
            filtered_lines.append(clean_line)

        lines = filtered_lines
        unique_lines = list(dict.fromkeys(lines))
        if len(unique_lines) < len(lines):
            errors.append("Usunięto powtarzające się podtematy.")

        final_subtopics = []
        has_error = False

        for line in unique_lines:
            if line != line.strip():
                errors.append(f"Nazwa podtematu zawiera białe znaki na początku lub końcu: '{line}'")
                has_error = True
                continue
            if not validate_latex(line, errors):
                errors.append(f"Błąd LaTeX w podtemacie: '{line}'")
                has_error = True
                continue
            final_subtopics.append(line)

        if has_error and not final_subtopics:
            errors.append("Wszystkie podtematy zostały odrzucone ze względu na błędy formatowania.")
            return old_subtopics

        filtered_subtopics = []
        subtopics_names = [s[0] for s in subtopics]

        for name in final_subtopics:
            if name not in subtopics_names:
                errors.append(f"Podtemat '{name}' nie znajduje się w liście subtopics.")
            else:
                filtered_subtopics.append(name)

        if not filtered_subtopics:
            return old_subtopics

        return filtered_subtopics

    except Exception as e:
        errors.append(f"Błąd nieoczekiwany podczas parsowania podtematów: {str(e)}")
        return old_subtopics

subtopics_prompt = """Jesteś ekspertem edukacyjnym specjalizującym się w tworzeniu kompletnej i logicznie uporządkowanej siatki tematycznej na poziomie Matura Rozszerzona.
Twoim zadaniem jest wygenerowanie szczegółowych, kompletnych i jednoznacznie zdefiniowanych podtematów dla:

Przedmiot: {$subject$}
Rozdział: {$section$}
Temat: {$topic$}

Aktualne podtematy to:
{$subtopics$}

Błędy formatowania:
{$errors$}

Wymagania dotyczące podtematów:
1. Podtematy muszą być w pełni zgodne z podstawą programową MEN dla liceum i technikum, uwzględniając wszystkie zagadnienia z poziomu Matura Rozszerzona, w sposób logiczny, spójny tematycznie i kompletny, tak aby cyklicznie obejmowały 100% treści danego tematu.
2. Lista podtematów musi pokrywać 100% zagadnień tematu w sposób spójny logicznie i tematycznie, bez powtórzeń i synonimów.
3. Format LaTeX:
   - Wzory inline w \( ... \) lub $ ... $,
   - Wzory blokowe (display) w \[ ... \] lub $$ ... $$,
   - Nie używaj środowisk takich jak align, equation, array, matrix, itp.
4. Każdy podtemat powinien być pojedynczą linią tekstu, bez znaków nowej linii, tabulatorów, podwójnych spacji ani białych znaków na początku i końcu.
5. Podtemat nie może zawierać znaku średnika ; w tekście tytułu.
6. Nazwa podtematu i liczba ważności muszą być oddzielone bezpośrednio znakiem średnika ; bez spacji po średniku bezpośrednio następuje liczba całkowita od 0 do 100, bez znaku procentu i bez spacji.

Uwaga:
- Start: i End: muszą być na osobnych liniach, bez spacji ani innych znaków przed lub po nich.
- Między Start: i End: może znajdować się tylko lista podtematów, każdy na osobnej linii, bez pustych linii, komentarzy i dodatkowych znaków.
- Nie wolno dodawać żadnego tekstu, spacji ani znaków specjalnych poza tym, co znajduje się między Start: i End:.
- Między Start: i End: musi być tylko lista podtematów, bez pustych linii i komentarzy.
- Nie dodawaj niczego poza tym blokiem.

Przykład poprawnej odpowiedzi:
Start:
Definicja równania kwadratowego \(ax^2 + bx + c = 0\);30
Postać ogólna równania kwadratowego $ax^2 + bx + c = 0$;25
Rozwiązywanie równań kwadratowych metodą faktoryzacji;45
End:
"""

task_prompt = """Jesteś nauczycielem przedmiotu {$subject$} w szkole średniej (liceum lub technikum). Twoim zadaniem jest wygenerowanie dokładnie jednego, poprawnie sformułowanego, zamkniętego zadania, które spełnia wszystkie poniższe wymagania dla:

Przedmiot: {$subject$}
Rozdział: {$section$}
Temat: {$topic$}
Poziom trudności: {$difficulty$}%
Próg: {$threshold$}%
Historia zadań:
{$tasks$}

Podtematy i poziom ich opanowania rozdzielone separatorem ; są:
{$subtopics$}

Aktualny tekst zadania:
{$text$}

Aktualne podtematy, które były ujęte w treści zadania:
{$outputSubtopics$}

Błędy formatowania:
{$errors$}

Wymagania dotyczące tekstu zadania:
1. Format LaTeX:
   - Wszystkie wzory inline muszą być w \( ... \) lub $ ... $.
   - Wszystkie wzory blokowe muszą być w \[ ... \] lub $$ ... $$.
   - Nie używaj środowisk takich jak align, equation, array, matrix, itp.
2. Zasady wyboru podtematów:
   - Wybierz podtematy zgodnie z hierarchią:
     1. Podtematy z opanowaniem niższym, ale bliskim progowi {$threshold$}%.
     2. Jeśli brak takich – wybierz te z 0% opanowaniem.
     3. Jeśli nadal brak – wybierz te minimalnie powyżej progu {$threshold$}%.
     4. Jeśli nadal brak – wybierz inne z najniższym poziomem opanowania.
   - Maksymalnie wybieraj możliwą liczbę podtematów spełniających powyższe kryteria, tak aby treść zadania obejmowała jak największy zakres materiału z wybranych podtematów.
3. Treść zadania:
   - Nie może zawierać odpowiedzi, rozwiązań, wariantów, komentarzy, podpowiedzi, tagów ani placeholderów.  
   - Nie dziel się na podpunkty ani nie używaj list – musi to być jedno spójne polecenie.
   - Nie powtarzaj żadnych konstrukcji z listy zadań
   {$tasks$}  
   - Stosuj różnorodne konstrukcje zdaniowe, unikaj powtarzających się schematów.  
   - Treść zadania musi być jednym spójnym zdaniem lub blokiem zdań, bez list i numeracji.  
   - Treść zadania musi być pytaniem zamkniętym, bez wariantów odpowiedzi w treści.  
4. Cel:
   - Wygeneruj jedno zadanie, które spełnia poziom trudności {$difficulty$}% i opiera się na najlepiej dobranych podtematach z listy
   {$subtopics$}  
   - Zadanie musi być unikalne względem historii zadań
   {$tasks$}
5. Podtematy w bloku subtopics:
   - W osobnym bloku po treści zadania wypisz tylko te podtematy z listy
   {$subtopics$}
   które zostały faktycznie użyte w treści zadania, bez procentów, zer, pustych linii ani innych znaków.  
   - Nie dopisuj żadnych podtematów, które nie wystąpiły w treści zadania.

Uwaga:
- Start: i End: muszą być na osobnych liniach, bez spacji ani innych znaków przed lub po nich.  
- Między Start: i End: musi znajdować się tekst zadania jako jeden nieprzerwany akapit.  
- Nie dodawaj niczego poza tym blokiem.  
- Między subtopicsStart: i subtopicsEnd: wypisz dokładnie wszystkie podtematy, które zostały użyte, bez pustych linii, procentów i komentarzy.

Przykład poprawnej odpowiedzi:
Start:
Treść zadania w jednym bloku, bez nowych linii, komentarzy ani dodatkowych znaków
End:
subtopicsStart:
Definicja równania kwadratowego \(ax^2 + bx + c = 0\)
Postać ogólna równania kwadratowego $ax^2 + bx + c = 0$
Rozwiązywanie równań kwadratowych metodą faktoryzacji
subtopicsEnd:
"""

solution_prompt = """Jesteś nauczycielem przedmiotu {$subject$} w szkole średniej (liceum lub technikum). Twoim zadaniem jest wygenerowanie dokładnie jednego, poprawnie sformułowanego, rozwiązania, które spełnia wszystkie poniższe wymagania dla:

Aktualny tekst zadania:
{$text$}

Aktualne rozwiązanie zadania:
{$solution$}

Błędy formatowania:
{$errors$}

Wymagania dotyczące tekstu zadania:
1. Format LaTeX:
   - Wszystkie wzory inline muszą być w \( ... \) lub $ ... $.
   - Wszystkie wzory blokowe muszą być w \[ ... \] lub $$ ... $$.
   - Nie używaj środowisk takich jak align, equation, array, matrix, itp.
2. Format odpowiedzi:
Zwróć dokładnie tekst rozwiązania:
- Bez cudzysłowów, komentarzy, nagłówków ani dodatkowych informacji oraz wstępów
3. Cel:
Wygenerowanie rozwiązania zadania, które:
- Spełnia prawidłowości dokładności według tekstu zadania:
{$text$}

Uwaga:
- Start: i End: muszą być na osobnych liniach, bez spacji ani innych znaków przed lub po nich.
- Między Start: i End: musi znajdować się rozwiązanie zadania.
- Nie wolno dodawać żadnego tekstu, spacji ani znaków specjalnych poza tym, co znajduje się między Start: i End:.
- Między Start: i End: musi być tylko rozwiązanie zadania, bez pustych linii i komentarzy.
- Nie dodawaj niczego poza tym blokiem.
- Tekst nie może zawierać cudzysłowów, dodatkowych nawiasów ani symboli spoza wzorów LaTeX.
- Rozwiązanie zadania powinno być przedstawione jako tekst ciągły, bez podziału na linie. Cała zawartość rozwiązania zadania powinna być podana jako jeden nieprzerwany akapit.

Przykład poprawnej odpowiedzi:
Start:
Rozwiązanie zadania w jednym bloku, bez nowych linii, komentarzy ani dodatkowych znaków
End:
"""

options_prompt = """Jesteś ekspertem edukacyjnym w tworzeniu wariantów odpowiedzi do zadań szkolnych (liceum/technikum). Wygeneruj dokładnie 4 warianty odpowiedzi.

Treść zadania:
{$text$}

Rozwiązanie zadania:
{$solution$}

Aktualne warianty:
{$options$}

Aktualny Indeks poprawnej odpowiedzi (0–3):
{$correctOptionIndex$}

Błędy formatowania:
{$errors$}

Wymagania:
1. Każdy wariant to **jedna linia**, max 200 znaków, bez nowych linii i dodatkowych spacji.
2. Wszystkie warianty unikalne; żaden nie powtarza innego ani rozwiązania.
3. Tylko jeden wariant poprawny zgodny w 100% z rozwiązaniem {$solution$}.
4. Pozostałe 3 warianty błędne, ale logiczne, typowe dla ucznia.
5. Nie używaj A), B), 1., -, numeracji ani innych znaków.
6. LaTeX: \( ... \) lub $ ... $ inline; \[ ... \] lub $$ ... $$ blok; zakazane środowiska: align, equation, array, matrix.
7. Po 4 wariantach podaj **tylko numer poprawnej odpowiedzi (0–3)** w nowej linii, bez znaków ani spacji.
8. Nie dodawaj komentarzy ani pustych linii.
9. Jeśli warianty i indeks już poprawne, zwróć je bez zmian.

Uwaga:
- Między Start: i End: znajdują się dokładnie 4 linie z wariantami odpowiedzi oraz jedna linia z numerem poprawnej odpowiedzi (0–3).
- Między Start: i End: musi znajdować się 4 warianty odpowiedzi a potem numer lub index prawidłowej odpowiedz (0, 1, 2 lub 3), każdy wariant oraz numer lub index prawidłowej odpowiedzi na osobnej linii, bez pustych linii.
- Nie wolno dodawać żadnego tekstu, spacji ani znaków specjalnych poza tym, co znajduje się między Start: i End:
- Warianty odpowiedzi nie mogą zawierać pustych linii pomiędzy sobą.
- Każdy wariant musi być unikalny, nie powtarzaj żadnego błędu ani parafrazy. Nie dziel odpowiedzi na kilka zdań – wszystko w jednej logicznej linii.
- Każda linia między Start: i End: może zawierać tylko pełny tekst wariantu odpowiedzi lub sam numer indeksu poprawnej odpowiedzi.
- Żaden wariant nie może być skrótem, urwanym fragmentem lub odpowiedzią pozbawioną sensu – wszystkie muszą być logicznie spójne z treścią zadania.
- Jeżeli aktualne warianty odpowiedzi oraz aktualny numer lub index prawidłowej odpowiedzi są już poprawne zgodnie z wymaganiami, zwróć dokładnie te same warianty odpowiedzi i ten sam numer lub index prawidłowej odpowiedzi, nie generując nowych.

Przykład poprawnej odpowiedzi:
Start:
Odejmując \(5\) od obu stron otrzymujemy \(2x=8\), a następnie dzieląc przez \(2\) otrzymujemy \(x=4\).
Odejmując \(5\) od obu stron otrzymujemy \(2x=10\), a następnie dzieląc przez $2$ otrzymujemy $x=5$.
Z równania \(2x+5=15\) dzielimy obie strony przez \(2\), otrzymując \(x+2{,}5=7{,}5\), a następnie odejmując \(2{,}5\) dostajemy \(x=5\).
Przenosząc \(5\) na prawą stronę ze zmianą znaku otrzymujemy \(2x=15+5\), a po podzieleniu przez \(2\) mamy \(x=10\).
1
End:
"""

closedSubtopics_prompt = """Jesteś ekspertem edukacji średniej (liceum lub technikum).
Twoim zadaniem jest ocenienie błędów według rozwiązaniu ucznia zadania: {$text$} z wariantami odpowiedzi
{$options$},
gdzie prawidłowy indeks odpowiedzi (0, 1, 2 lub 3) to {$correctOptionIndex$},
dla przedmiotu {$subject$}, rozdziału {$section$}, tematu {$topic$}, o poziomie trudności {$difficulty$}%.

Poprawnym rozwiązaniem zadania jest {$solution$}.

Określ, które podtematy
{$subtopics$}
jako {$currentsubtopics$} były uwzględnione w treści zadania {$text$}
Uwzględniaj wyłącznie te podtematy, które mają wyraźne odniesienie w treści zadania. Inne podtematy pomijaj całkowicie.

Najważniejsza jest ocena wyjaśnienia ucznia {$userSolution$}. 
Jeżeli {$userOptionIndex$} != {$correctOptionIndex$}, ustaw minimalny procent błędu 50% dla wszystkich podtematów z {$currentsubtopics$} obecnych w treści zadania, nawet jeśli wyjaśnienie ucznia {$userSolution$} jest częściowo poprawne.
Jeżeli {$userOptionIndex$} == {$correctOptionIndex$}, ale {$userSolution$} nie odnosi się do żadnych podtematów z {$currentsubtopics$}, ustaw procent błędu 100% dla wszystkich tych podtematów.

Weź wybrany przez ucznia numer odpowiedzi (0, 1, 2 lub 3) — {$userOptionIndex$} oraz jego wyjaśnienia {$userSolution$},
aby określić błędy rozwiązania w procentach dla każdego podtematu z {$currentsubtopics$},
które bezpośrednio występują w treści zadania {$text$}.

Porównaj wynik z poprzednią oceną błędów:
{$outputSubtopics$}

Błędy danych do korekcji formatowania:
{$errors$}

Wymagania dotyczące formatu prezentowanych danych:
1. Dla każdej podtematy z {$currentsubtopics$}: 
- 0% jeśli rozwiązanie poprawnie obejmuje podtemat, 
- minimalny 50% jeśli wariant odpowiedzi jest błędny, 
- 100% jeśli podtemat nie został uwzględniony w wyjaśnieniu ucznia.
2. Procent błędu dla podtematów powinien wskazywać, jak bardzo uczeń popełnił błąd — im większa pomyłka, tym wyższy procent. Jeśli podtemat został rozwiązany poprawnie, nie dodawaj go do listy.
3. Uwzględnij, czy wyjaśnienie ucznia {$userSolution$} logicznie prowadzi do poprawnego wyniku, oraz oceń błędy na podstawie argumentacji, a nie tylko wybranego wariantu odpowiedzi {$userOptionIndex$}.
4. Jeżeli wyjaśnienie jest poprawne i w pełni odpowiada podtematom, nie zwiększaj procentu błędu do maksimum tylko z powodu błędnie wybranego wariantu odpowiedzi.
5. Procent błędu powinien odzwierciedlać wyłącznie logiczne niedociągnięcia w wyjaśnieniu, a nie sam fakt niepoprawnego wyboru odpowiedzi.
6. Ocena procentowa dotyczy wyłącznie podtematów występujących w treści zadania. Podtematy niewystępujące w treści zadania nie są uwzględniane w ocenie ani w liście wynikowej. 
7. Format LaTeX:
   - Wszystkie wzory inline muszą być w \( ... \) lub $ ... $.
   - Wszystkie wzory blokowe muszą być w \[ ... \] lub $$ ... $$.
   - Nie używaj środowisk takich jak align, equation, array, matrix, itp.
8. Podtematy i procent błędów rozdzielone separatorem ; powinny występować w liście podtematów
{$subtopics$}
oraz nie wolno dodawać żadnych nowych podtematów, nawet jeśli pojawiają się w treści rozwiązania ucznia. Należy używać tylko według listy podtematów.
Uwzględniaj tylko te podtematy, które występują dokładnie w treści zadania {$text$}. Jeśli podtemat nie występuje w treści zadania, nie dodawaj go do listy wcale.
9. Podtemat nie może zawierać znaku średnika ; w tekście tytułu.
10. Nazwa podtematu i procent błędów muszą być oddzielone bezpośrednio znakiem średnika ; bez spacji po średniku bezpośrednio następuje liczba całkowita od 0 do 100, bez znaku procentu i bez spacji.
11. Procent opanowania podtematów musi prawidłowo wskazywać na procent błędów danego podtematu.

Uwaga:
- Start: i End: muszą być na osobnych liniach, bez spacji ani innych znaków przed lub po nich.
- Między Start: i End: może znajdować się tylko lista podtematów, każdy na osobnej linii, bez pustych linii, komentarzy i dodatkowych znaków.
- Nie wolno dodawać żadnego tekstu, spacji ani znaków specjalnych poza tym, co znajduje się między Start: i End:.
- Między Start: i End: musi być tylko lista podtematów, bez pustych linii i komentarzy.
- Nie dodawaj niczego poza tym blokiem.

Przykład poprawnej odpowiedzi:
Start:
Definicja równania kwadratowego \(ax^2 + bx + c = 0\);30
Postać ogólna równania kwadratowego $ax^2 + bx + c = 0$;25
Rozwiązywanie równań kwadratowych metodą faktoryzacji;45
End:
"""

text_language_prompt = """Jesteś nauczycielem przedmiotu {$subject$} w szkole średniej (liceum lub technikum). Napisz krótkie opowiadanie w języku angielskim, które ma dokładnie 2 akapity. W tekście użyj jak najwięcej elementów gramatyki z listy podtematów, które mają najwyższy procent opnowania oraz pełne tłumaczenie w języku polskim.

Przedmiot: {$subject$}
Rozdział: {$section$}
Temat: {$topic$}
Poziom trudności: {$difficulty$}%

- W tekście opowiadania obowiązkowo użyj elementów gramatyki z listy podtematów, które podano poniżej. Priorytetowo zastosuj te, które mają najwyższy procent opanowania. Procenty są podane po średniku.
Podtematy gramatyki i poziom ich opanowania rozdzielone separatorem ; są:
{$subtopics$}

Aktualny tekst opowiadania w języku angielskim:
{$text$}

Aktualne tłumaczenie tekstu opowiadania w języku polskim:
{$translate$}

Błędy formatowania:
{$errors$}

Uwaga:
- Start: i End: muszą być na osobnych liniach, bez spacji ani innych znaków przed lub po nich.  
- Między Start: i End: musi znajdować się tekst opowiadania w języku angielskim jako dwa akapity napisane ciągłym tekstem w jednej linii, bez znaków nowej linii, pustych wierszy czy jakichkolwiek separatorów.
- Nie dodawaj niczego poza tym blokiem.  
- Między translateStart: i translateEnd: wypisz pełne tłumaczenie w języku polskim w tej samej strukturze: dwa akapity napisane ciągłym tekstem w jednej linii, bez znaków nowej linii, pustych wierszy czy separatorów.
- Nie dodawaj ani pojaśnień, pustych linii, komentarzy
- Nie powtarzaj informacji o przedmiocie, rozdziale, temacie ani błędach w odpowiedzi.
- Tekst w blokach Start, End, translateStart i translateEnd musi być całkowicie w jednej linii, bez nowej linii, pustych wierszy, separatorów lub innych dodatkowych znaków. 
- Nie używaj znaków końca linii w środku tekstu; cały tekst powinien być ciągły.

Przykład poprawnej odpowiedzi:
Start:
Yesterday I woke up early because I wanted to go to the park. The sun was shining, and the birds were singing. While I was walking down the street, I saw my friend Anna. She was riding her bike and waving at me. We talked for a few minutes about what we did last weekend. I told her I visited my grandparents, and she said she was studying for an exam all Saturday. Suddenly, it started raining, so we ran to the nearest café and had some coffee together.
End:
translateStart:
Wczoraj obudziłem się wcześnie, ponieważ chciałem pójść do parku. Słońce świeciło, a ptaki śpiewały. Kiedy szedłem ulicą, zobaczyłem moją przyjaciółkę Annę. Jechała na rowerze i machała do mnie. Rozmawialiśmy przez kilka minut o tym, co robiliśmy w zeszły weekend. Powiedziałem jej, że odwiedziłem dziadków, a ona powiedziała, że uczyła się do egzaminu przez całą sobotę. Nagle zaczęło padać, więc pobiegliśmy do najbliższej kawiarni i wypiliśmy razem kawę.
translateEnd:
"""

text_subtasks_prompt = """Jesteś nauczycielem przedmiotu {$subject$} w szkole średniej (liceum lub technikum). Twoim zadaniem jest wygenerowanie listę pytań sprawdzających zrozumienie tekstu opowiadania w języku angielskim, które znajdują się w polu {$text$}.

Przedmiot: {$subject$}
Rozdział: {$section$}
Temat: {$topic$}
Poziom trudności: {$difficulty$}%

Tekst opowiadania w języku angielskim:
{$text$}

Aktualna lista pytań:
{$questions$}

Błędy formatowania:
{$errors$}

Wymagania dotyczące listy pytań:
1. Każde pytanie powinno dotyczyć wyłącznie treści tekstu opowiadania i weryfikować zrozumienie jego szczegółów, wydarzeń i kontekstu.
2. Każde pytanie powinno być pojedynczą linią tekstu, bez znaków nowej linii, tabulatorów, podwójnych spacji ani białych znaków na początku i końcu.
3. Nie twórz odpowiedzi ani opcji ABCD/1234, podawaj wyłącznie treść pytań.
4. Zachowaj sens i kolejność wydarzeń z tekstu.

Uwaga:
- Start: i End: muszą być na osobnych liniach, bez spacji ani innych znaków przed lub po nich.
- Między Start: i End: może znajdować się tylko lista pytań, każdy na osobnej linii, bez pustych linii, komentarzy i dodatkowych znaków.
- Nie wolno dodawać żadnego tekstu, spacji ani znaków specjalnych poza tym, co znajduje się między Start: i End:.

Przykład poprawnej odpowiedzi:
Start:
What did the main character do first in the morning?
Who did they meet while walking in the park?
What activity did they do together after meeting?
End:
"""