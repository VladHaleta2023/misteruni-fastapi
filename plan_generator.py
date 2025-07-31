import re
import logging
from typing import List, Dict
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)

class Section(BaseModel):
    section: str
    topics: List[str]

class PlanResponse(BaseModel):
    sections: List[Section]


def remove_duplicates(sections: List[Dict[str, List[str]]]) -> List[Dict[str, List[str]]]:
    for section in sections:
        seen = set()
        unique_topics = []
        for topic in section["topics"]:
            normalized = topic.strip()
            if normalized not in seen:
                unique_topics.append(normalized)
                seen.add(normalized)
        section["topics"] = unique_topics
    return sections

def parse_plan(text: str) -> List[Dict[str, List[str]]]:
    sections = []
    current_section = None
    current_topic_lines = []
    current_topic_id = None

    lines = text.strip().splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        section_match = re.match(r"^(\d+)\.\s+(.+)", line)
        topic_match = re.match(r"^(\d+\.\d+)\s+(.+)", line)

        if section_match and not topic_match:
            if current_topic_lines and current_section is not None:
                full_topic = "\n".join(current_topic_lines).strip()
                current_section["topics"].append(full_topic)
                current_topic_lines = []

            current_section = {"section": section_match.group(2).strip(), "topics": []}
            sections.append(current_section)
            current_topic_id = None

        elif topic_match:
            if current_topic_lines and current_section is not None:
                full_topic = "\n".join(current_topic_lines).strip()
                current_section["topics"].append(full_topic)

            current_topic_id = topic_match.group(1)
            first_line = topic_match.group(2).strip()

            current_topic_lines = [first_line]

        elif current_section is not None and current_topic_id:
            current_topic_lines.append(line)

    if current_topic_lines and current_section is not None:
        full_topic = "\n".join(current_topic_lines).strip()
        current_section["topics"].append(full_topic)

    return remove_duplicates(sections)

def full_plan_generate(prompt: str) -> PlanResponse:
    sections = parse_plan(prompt)
    return PlanResponse(sections=sections)


