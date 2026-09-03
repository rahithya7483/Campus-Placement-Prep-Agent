import re
import sys
import os
from pydantic import BaseModel, Field
from typing import List, Literal

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from llm_client import get_llm_call


class MockInterviewQuestion(BaseModel):
    question: str
    model_answer_points: List[str] = Field(min_length=2, max_length=6)
    difficulty: Literal["easy", "medium", "hard"]
    topic: str


FEW_SHOT_EXAMPLES = """
Example 1:
Role: Data Engineer, Topic: SQL
{"question": "Write a query to find the second-highest salary in an Employees table.",
 "model_answer_points": ["Use ORDER BY + LIMIT/OFFSET, or a subquery with MAX()",
   "Handle ties/duplicates explicitly", "Mention window functions (DENSE_RANK) as an alternative"],
 "difficulty": "medium", "topic": "SQL"}

Example 2:
Role: Software Development, Topic: Behavioral
{"question": "Tell me about a time you disagreed with a teammate on a technical decision.",
 "model_answer_points": ["Use STAR structure", "Focus on the resolution, not the conflict",
   "Show you listened to the other perspective"],
 "difficulty": "easy", "topic": "Behavioral"}
"""


def generate_question(role: str, topic: str, llm_call) -> MockInterviewQuestion:
    prompt = f"""You generate mock interview questions for engineering students.
Follow the style of these examples exactly:
{FEW_SHOT_EXAMPLES}

Now generate ONE question for:
Role: {role}
Topic: {topic}

Respond ONLY with valid JSON matching this schema, nothing else:
{{"question": str, "model_answer_points": [str] (2-6 items), "difficulty": "easy"|"medium"|"hard", "topic": str}}
"""
    raw = llm_call(prompt)
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return MockInterviewQuestion.model_validate_json(raw)


if __name__ == "__main__":
    result = generate_question("Data Engineer", "DSA", get_llm_call())
    print(result.model_dump_json(indent=2))