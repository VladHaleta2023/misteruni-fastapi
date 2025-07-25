import re
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel
from configAI import client, FORMAT_INSTRUCTION, top_plan_models

logging.basicConfig(level=logging.INFO)

class Section(BaseModel):
    section: str
    topics: List[str]

class PlanResponse(BaseModel):
    sections: List[Section]


def remove_duplicates(sections: List[Dict[str, List[str]]]) -> List[Section]:
    seen_sections = set()
    clean_sections = []
    for section in sections:
        name = section["section"]
        if name in seen_sections:
            continue
        seen_sections.add(name)
        seen_topics = set()
        topics = []
        for t in section["topics"]:
            if t not in seen_topics:
                seen_topics.add(t)
                topics.append(t)
        clean_sections.append(Section(section=name, topics=topics))
    return clean_sections

def parse_plan(text: str) -> List[Section]:
    sections: List[Dict[str, List[str]]] = []
    current: Optional[Dict[str, List[str]]] = None

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        section_match = re.match(r"^(\d+)\.\s*(.+)", line)
        topic_match = re.match(r"^(\d+\.\d+)\s*(.+)", line)

        if section_match and not topic_match:
            current = {"section": section_match.group(2).strip(), "topics": []}
            sections.append(current)
        elif topic_match and current:
            current["topics"].append(topic_match.group(2).strip())

    return remove_duplicates(sections)


def full_plan_generate(prompt: str) -> PlanResponse:
    sections = parse_plan(prompt)
    return PlanResponse(sections=sections)


