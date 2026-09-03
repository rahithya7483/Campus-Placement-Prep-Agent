import sys
import os
import re
from pydantic import BaseModel, Field
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from llm_client import get_llm_call


class ResumeScore(BaseModel):
    score: int = Field(ge=0, le=100)
    missing_keywords: List[str]
    formatting_issues: List[str]
    top_fixes: List[str] = Field(max_length=3)

class InsufficientResumeError(ValueError):
    """Raised when resume_text is empty, a placeholder, or too short to be a real resume."""
    pass


_PLACEHOLDER_STRINGS = {"n/a", "na", "none", "nil", "-", "test", "resume", "my resume", "asdf"}


def is_resume_text_insufficient(resume_text: str) -> bool:
    text = (resume_text or "").strip()
    if not text:
        return True
    if text.lower() in _PLACEHOLDER_STRINGS:
        return True
    # a real resume - even a thin one - has far more than a few words:
    # name, education line, a handful of skills, at least one project/experience bullet
    if len(text.split()) < 20 or len(text) < 100:
        return True
    return False


def score_resume(resume_text: str, job_description: str, llm_call) -> ResumeScore:
    if is_resume_text_insufficient(resume_text):
        raise InsufficientResumeError(
            "That doesn't look like a full resume - please paste your actual resume text "
            "(education, skills, and at least one project or experience entry) so I can "
            "score it properly."
        )

    if job_description and job_description.strip():
        prompt = JD_MATCH_PROMPT.format(resume_text=resume_text, job_description=job_description)
    else:
        prompt = GENERAL_ATS_PROMPT.format(resume_text=resume_text)

    raw = llm_call(prompt)
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return ResumeScore.model_validate_json(raw)


JD_MATCH_PROMPT = """You are an ATS resume screener. Compare this resume against
the job description. Respond ONLY with valid JSON matching this schema,
no extra text:
{{"score": int (0-100), "missing_keywords": [str], "formatting_issues": [str], "top_fixes": [str, max 3]}}

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}
"""

GENERAL_ATS_PROMPT = """You are an ATS resume screener. No specific job description was
provided, so evaluate this resume purely on general ATS and resume-quality best practices:
quantified impact/metrics in bullet points, strong action verbs, standard section headers
(Education, Experience, Skills, Projects), consistent formatting that ATS parsers can read
(no tables/graphics/columns), and keyword density relevant to the resume's own apparent
target role. Respond ONLY with valid JSON matching this schema, no extra text:
{{"score": int (0-100), "missing_keywords": [str] (skills/keywords typically expected for
the resume's apparent target role that are missing), "formatting_issues": [str], "top_fixes": [str, max 3]}}

RESUME:
{resume_text}
"""


def score_resume(resume_text: str, job_description: str, llm_call) -> ResumeScore:
    if job_description and job_description.strip():
        prompt = JD_MATCH_PROMPT.format(resume_text=resume_text, job_description=job_description)
    else:
        prompt = GENERAL_ATS_PROMPT.format(resume_text=resume_text)

    raw = llm_call(prompt)
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return ResumeScore.model_validate_json(raw)


if __name__ == "__main__":
    sample_resume = "B.Tech Data Science student. Skills: Python, SQL, pandas, scikit-learn. Built a placement stats analysis project."

    print("--- With JD ---")
    print(score_resume(sample_resume, "Looking for a Data Engineer with Python, SQL, Docker, CI/CD.", get_llm_call()).model_dump_json(indent=2))

    print("--- Without JD (general ATS check) ---")
    print(score_resume(sample_resume, "", get_llm_call()).model_dump_json(indent=2))