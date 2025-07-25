from g4f.client import Client

client = Client()

FORMAT_INSTRUCTION = """
Odpowiedz WYŁĄCZNIE czystym tekstem w dokładnym formacie, bez żadnych odstępstw:

1. Pełna nazwa działu  
  1.1 Pełna nazwa tematu  
  1.2 Pełna nazwa tematu  
  ...  
2. Pełna nazwa kolejnego działu  
  2.1 Pełna nazwa tematu  
  2.2 Pełna nazwa tematu  
  ...

NIE dodawaj niczego poza tym formatem.  
NIE dodawaj żadnych komentarzy, wstępów, podsumowań, ani pustych linii.  
NIE dodawaj dodatkowych spacji ani znaków poza tymi, które są w przykładzie powyżej.  
KAŻDY dział i KAŻDY temat MUSI pojawić się dokładnie JEDEN RAZ.  
ABSOLUTNIE ŻADNYCH powtórzeń, duplikatów ani parafraz.  
NIE zmieniaj numeracji ani formatowania — musi być dokładnie tak, jak pokazano (cyfry, kropki, spacje).  
ZACHOWAJ dokładną kolejność działów i tematów.  
Jeśli coś nie pasuje, zwiórz błąd lub NIE generuj odpowiedzi.  
Ściśle przestrzegaj tego formatu i NIC więcej.  
"""

top_plan_models = [
    "claude-3-opus",
    "gpt-4.5",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-4o",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "wizardlm-2-8x22b",
    "llama-3.1-405b",
    "qwen-3-235b",
    "phi-4",
    "llama-3.3-70b",
    "llama-3.2-90b",
    "qwen-2.5-max",
    "llama-3.1-70b",
    "gemma-3-27b",
    "mistral-7b",
    "wizardlm-2-7b",
    "qwen-2-72b",
    "gemma-3-12b",
    "mistral-nemo",
    "mistral-small-24b",
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "gpt-4.1-nano",
    "phi-3.5-mini",
    "o4-mini-high",
    "o4-mini",
    "o3-mini-high",
    "o3-mini",
    "mistral-small-3.1-24b"
]