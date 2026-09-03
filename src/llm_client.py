import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
MOCK_LLM = os.getenv("MOCK_LLM", "true").lower() == "true"
GROQ_MODEL = "openai/gpt-oss-20b"

def _real_groq_call(prompt: str) -> str:
    client = Groq(api_key = os.getenv("GROQ_API"))
    response = client.chat.completions.create(
        model = GROQ_MODEL, messages = [{"role":"user", "content": prompt}],
        temperature = 0.3
    )
    return response.choices[0].message.content

def _mock_call(prompt: str) -> str:
    """
    Very rough mock: looks at the prompt to guess which tool called it,
    so both tools get a plausible-shaped response without hitting any API.
    """
    if "Answer the user's question using ONLY the conversation history" in prompt:
        return "Mock answer synthesized from conversation history."
    if "ATS resume screener" in prompt:
        return json.dumps({
            "score": 72,
            "missing_keywords": ["Docker", "CI/CD"],
            "formatting_issues": ["No quantified impact in bullet points"],
            "top_fixes": ["Add metrics to project bullets", "Include a skills section", "Match JD keywords more closely"],
        })
    if msg.strip() in ("hi", "hello", "hey", "hii") or "what can you do" in msg or "who are you" in msg:
        return json.dumps({"intent": "general_chat", "params": {"subtype": "greeting"}})
    if any(w in msg for w in ["thank", "thanks", "bye", "goodbye", "see you", "cya"]):
        return json.dumps({"intent": "general_chat", "params": {"subtype": "closing"}})
    if "mock interview questions" in prompt:
        return json.dumps({
            "question": "Explain the difference between a HashMap and a TreeMap.",
            "model_answer_points": [
                "HashMap: O(1) average lookup, no ordering guarantee",
                "TreeMap: O(log n) lookup, keys stored in sorted order",
            ],
            "difficulty": "medium",
            "topic": "DSA",
        })
    if "required first identifying a company from placement" in prompt:
        return "Mock compound answer: identified company + its interview notes synthesized together."
    if "Classify the user's message" in prompt:
        msg = prompt.split("User message:")[-1].split("JSON:")[0].strip().lower()
        if "resume" in msg:
            return json.dumps({"intent": "resume_score", "params": {"resume_text": "", "job_description": ""}})
        if "interview question" in msg or "mock interview" in msg or "practice question" in msg or "quiz me" in msg:
            return json.dumps({"intent": "mock_interview", "params": {"role": "Software Engineer", "topic": "general"}})
        if any(w in msg for w in ["highest", "compare", "how many times", "total", "package", "pay", "package offered"]):
            return json.dumps({"intent": "placement_stats", "params": {"query": msg}})
        return json.dumps({"intent": "placement_notes", "params": {"query": msg}})

    if "Answer the user's question using ONLY the context" in prompt:
        return "Mock answer synthesized from the retrieved context chunks."

    if "You write SQLite SELECT queries" in prompt:
        return "SELECT clean_company, MAX(ctc_lpa) as max_ctc FROM placement_stats WHERE year = '2016-17'"

    if "SQL query result" in prompt:
        return "Mock stats answer synthesized from the SQL query result."

    return json.dumps({"error": "mock_call: unrecognized prompt type"})

def get_llm_call():
    """Returns a function(prompt: str) -> str, real or mock depending on MOCK_LLM."""
    if MOCK_LLM:
        return _mock_call
    if not os.getenv("GROQ_API"):
        raise RuntimeError("MOCK_LLM=false but GROQ_API_KEY is not set in .env")
    return _real_groq_call

if __name__ == "__main__":
    llm_call = get_llm_call()
    print(f"MOCK_LLM = {MOCK_LLM}")
    print(llm_call("test prompt containing mock interview questions"))