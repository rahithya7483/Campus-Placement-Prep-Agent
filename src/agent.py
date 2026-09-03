import sys
import os
import json
import time
import operator
from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

sys.path.append(os.path.dirname(__file__))
from llm_client import get_llm_call
from tools.resume_scorer import score_resume, InsufficientResumeError
from tools.mock_interview import generate_question
from tools.placement_rag import search_placement_data
from tools.placement_stats import answer_stats_question, query_placement_stats, is_effectively_empty

MAX_RETRIES = 3
MAX_HISTORY_TURNS = 6  # how many past (user, assistant) turns to feed back into prompts


class AgentState(TypedDict):
    user_message: str
    intent: Optional[str]
    params: dict
    tool_output: Optional[dict]
    final_response: Optional[str]
    error: Optional[str]
    history: Annotated[list, operator.add]  # reducer: returned values are APPENDED, not replaced


def call_llm_with_retry(prompt: str) -> str:
    llm_call = get_llm_call()
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return llm_call(prompt)
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)  # 1s, 2s, 4s
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts: {last_err}")


def format_history(history: list, max_turns: int | None = MAX_HISTORY_TURNS) -> str:
    if not history:
        return "(no prior conversation)"
    if max_turns is None:
        recent = history
    else:
        recent = history[-(max_turns * 2):]
    lines = [f"{turn['role'].capitalize()}: {turn['content']}" for turn in recent]
    return "\n".join(lines)

CLASSIFY_PROMPT = """Classify the user's message into exactly one intent and extract parameters.
Respond ONLY with valid JSON, no extra text.

Intents:
- "general_chat": user is engaging in small talk rather than a real placements/resume/interview
  question - either opening (greeting, "what can you do") or closing (thanks, goodbye)
  params: {{"subtype": "greeting"|"closing"}}
- "resume_score": - "resume_score": user wants their resume evaluated - against a job description if one is
  given in the message, otherwise evaluate it on general ATS/resume-quality best practices.
  If the user refers to a resume already shared earlier ("that resume", "my resume", "score
  it for a fresher role") without repasting it, find the actual resume text in the most
  recent message in the conversation history that contains it, and copy it into resume_text
  below - do not treat a missing resume in the CURRENT message as missing information if it
  exists earlier in the history.
  params: {{"resume_text": str, "job_description": str}}  # job_description can be ""
  Only use intent "placement_notes" for missing information if the resume text cannot be found
ANYWHERE - not in the current message, and not in the conversation history above. Check
history before concluding information is missing.
- "mock_interview": user wants you to GENERATE a new practice question for them to
  attempt - no specific company is being asked about, just a topic/role
  params: {{"role": str, "topic": str}}
- "placement_notes": user asks a QUALITATIVE question about a specific company -
  interview process, rounds, sample questions asked, role type
  params: {{"query": str}}
- "placement_stats": user asks a NUMERIC/AGGREGATE/COMPARISON question - packages,
  headcounts, "highest", "compare X and Y", "how many times", totals, superlatives
  params: {{"query": str}}
- "compound_query": the question requires BOTH a stats lookup (to identify a company via a
  numeric criterion like "highest package", "most students selected") AND then asking about
  that company's interview process/notes - i.e. it chains stats -> notes in one question
  params: {{"stats_query": str (the part that identifies the company via numbers),
            "notes_query": str (what they want to know about that company, e.g. "interview process")}}
- "conversation_context": user is asking about the conversation itself (e.g. "summarize
   what I've asked", "what did I ask just now") OR asking a follow-up question that depends
   on something discussed earlier in this session (e.g. "is this person able to get hired?"
   after a resume was already pasted and scored) rather than starting a new request
   params: {{}}

Recent conversation (use this to resolve follow-ups like "give me a harder one" or
"what about Infosys too" - rewrite the params so they're self-contained, not
dependent on the history):
{history}

Examples:
"hi" -> {{"intent": "general_chat", "params": {{"subtype": "greeting"}}}}
"what can you do?" -> {{"intent": "general_chat", "params": {{"subtype": "greeting"}}}}
"thank you" -> {{"intent": "general_chat", "params": {{"subtype": "closing"}}}}
"bye" -> {{"intent": "general_chat", "params": {{"subtype": "closing"}}}}
"give me a mock interview question on SQL" -> {{"intent": "mock_interview", "params": {{"role": "Software Engineer", "topic": "SQL"}}}}
"what's the interview process for TCS?" -> {{"intent": "placement_notes", "params": {{"query": "TCS interview process"}}}}
"compare TCS and Infosys packages" -> {{"intent": "placement_stats", "params": {{"query": "compare TCS and Infosys packages"}}}}
"what was the highest package in 2016-17?" -> {{"intent": "placement_stats", "params": {{"query": "highest package in 2016-17"}}}}
# (illustrative - actual history will contain the real pasted resume text)
"give that resume an ATS score for a fresher role" (resume was pasted 2 turns earlier)
-> {{"intent": "resume_score", "params": {{"resume_text": "<the actual resume text copied from history>", "job_description": ""}}}}
"summarize what I've asked in this session" -> {{"intent": "conversation_context", "params": {{}}}}
"what's the interview process of the company that paid the highest package in 2024-25?"
-> {{"intent": "compound_query", "params": {{"stats_query": "which company paid the highest package in 2024-25", "notes_query": "interview process"}}}}
"what did I ask just now" -> {{"intent": "conversation_context", "params": {{}}}}
"is this person able to get hired?" (after a resume was discussed earlier) -> {{"intent": "conversation_context", "params": {{}}}}

If required info is missing (e.g. resume_score without an actual resume pasted in),
use intent "placement_notes" with the user's message as the query instead - don't guess.

User message: {message}

JSON:
"""


def classify_intent(state: AgentState) -> dict:
    history_text = format_history(state.get("history", []))
    prompt = CLASSIFY_PROMPT.format(history=history_text, message=state["user_message"])
    try:
        raw = call_llm_with_retry(prompt)
        parsed = json.loads(raw)
        intent = parsed["intent"]
        params = parsed.get("params", {})
        error = None
    except Exception as e:
        intent = "placement_notes"
        params = {"query": state["user_message"]}
        error = f"classify_intent fallback: {e}"

    return {
        "intent": intent,
        "params": params,
        "error": error,
        "history": [{"role": "user", "content": state["user_message"]}],
    }

def run_resume_tool(state: AgentState) -> dict:
    try:
        result = score_resume(
            state["params"].get("resume_text", ""),
            state["params"].get("job_description", ""),
            get_llm_call(),
        )
        return {"tool_output": result.model_dump()}
    except InsufficientResumeError as e:
        return {"tool_output": {"answer": str(e)}}
    except Exception as e:
        return {"tool_output": None, "error": str(e)}


def run_interview_tool(state: AgentState) -> dict:
    try:
        result = generate_question(
            state["params"].get("role", "Software Engineer"),
            state["params"].get("topic", "general"),
            get_llm_call(),
        )
        return {"tool_output": result.model_dump()}
    except Exception as e:
        return {"tool_output": None, "error": str(e)}


def run_notes_rag_tool(state: AgentState) -> dict:
    try:
        query = state["params"].get("query", state["user_message"])
        chunks = search_placement_data(query)
        context = "\n".join(f"- {c['text']}" for c in chunks)
        synth_prompt = f"""Answer the user's question using ONLY the context below.
If the context doesn't contain the answer, say so honestly - don't guess.

Context:
{context}

Question: {state['user_message']}

Answer:
"""
        answer = call_llm_with_retry(synth_prompt)
        return {"tool_output": {"answer": answer, "sources": chunks}}
    except Exception as e:
        return {"tool_output": None, "error": str(e)}


def run_stats_tool(state: AgentState) -> dict:
    try:
        query = state["params"].get("query", state["user_message"])
        result = answer_stats_question(query, get_llm_call())
        return {"tool_output": result}
    except Exception as e:
        return {"tool_output": None, "error": str(e)}

def run_context_tool(state: AgentState) -> dict:
    try:
        history_text = format_history(state.get("history", []), max_turns=None)
        prompt = f"""Answer the user's question using ONLY the conversation history below.
This may be a request to summarize the conversation, recall an earlier message, or ask a
follow-up about something discussed earlier (e.g. a resume that was pasted earlier, along
with its ATS score). If the history doesn't contain enough information to answer, say so
honestly - don't guess.

Conversation history:
{history_text}

User's question: {state['user_message']}

Answer:
"""
        answer = call_llm_with_retry(prompt)
        return {"tool_output": {"answer": answer}}
    except Exception as e:
        return {"tool_output": None, "error": str(e)}

def run_compound_tool(state: AgentState) -> dict:
    try:
        stats_query = state["params"].get("stats_query", "")
        notes_query = state["params"].get("notes_query", "")

        stats_result = query_placement_stats(stats_query, get_llm_call())
        if stats_result["error"] or is_effectively_empty(stats_result["rows"]):
            return {"tool_output": {"answer": "I couldn't determine the company from the stats data for that question."}}

        first_row = stats_result["rows"][0]
        company = first_row.get("clean_company")
        if not company:
            return {"tool_output": {"answer": "I found matching data but couldn't identify a specific company from it."}}

        chunks = search_placement_data(f"{company} {notes_query}")
        if not chunks:
            return {"tool_output": {"answer": f"{company} is the company that matches your stats question, but I don't have interview/notes data recorded for {company}."}}

        context = "\n".join(f"- {c['text']}" for c in chunks)
        synth_prompt = f"""The user's question required first identifying a company from placement
statistics, and then finding notes about that specific company.

Identified company: {company}
Stats basis (question -> matching rows): {stats_query} -> {json.dumps(stats_result['rows'])}

Relevant notes about {company}:
{context}

Answer the user's original question below using ONLY the information above. If the notes don't
actually cover what they asked, say so honestly - don't guess.

Original question: {state['user_message']}

Answer:
"""
        answer = call_llm_with_retry(synth_prompt)
        return {"tool_output": {"answer": answer}}
    except Exception as e:
        return {"tool_output": None, "error": str(e)}

    
GENERAL_CHAT_GREETING = (
    "Hey! I'm your Campus Placement Prep Agent. I can only help with:\n"
    "- Visited Companies' interview processes & sample questions (e.g. \"What's the interview process for TCS?\")\n"
    "- College Placement stats (e.g. \"What was the highest package in 2024-25?\")\n"
    "- Mock interview questions (e.g. \"Give me a mock interview question on DSA\")\n"
    "- Resume ATS scoring - just paste your resume text (and a job description, if you have one)\n\n"
    "What would you like to start with?"
)

GENERAL_CHAT_CLOSING = "You're welcome! Good luck with your placements - come back anytime you need more help."

def run_general_chat_tool(state: AgentState) -> dict:
    subtype = state["params"].get("subtype", "greeting")
    answer = GENERAL_CHAT_CLOSING if subtype == "closing" else GENERAL_CHAT_GREETING
    return {"tool_output": {"answer": answer}}

def format_response(state: AgentState) -> dict:
    tool_output = state.get("tool_output")
    intent = state.get("intent")
    if tool_output is None:
        final = "Sorry, something went wrong processing that. Please try rephrasing your question."
    elif intent in ("placement_notes", "placement_stats", "conversation_context", "general_chat", "compound_query"):
        final = tool_output.get("answer", "I couldn't find an answer.")
    elif intent == "mock_interview":
        pts = "\n".join(f"- {p}" for p in tool_output.get("model_answer_points", []))
        final = f"**{tool_output.get('topic')}** ({tool_output.get('difficulty')})\n\n{tool_output.get('question')}\n\nKey points to cover:\n{pts}"
    elif intent == "resume_score":
        if "score" in tool_output:
            fixes = "\n".join(f"- {f}" for f in tool_output.get("top_fixes", []))
            missing = ", ".join(tool_output.get("missing_keywords", [])) or "none"
            final = f"**ATS Score: {tool_output.get('score')}/100**\n\nMissing keywords: {missing}\nTop fixes:\n{fixes}"
        else:
            final = tool_output.get("answer", "I couldn't score that.")
    else:
        final = json.dumps(tool_output, indent=2)  # fallback
    return {"final_response": final, "history": [{"role": "assistant", "content": final}]}


def route_by_intent(state: AgentState) -> str:
    return state["intent"]


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("resume_score", run_resume_tool)
    graph.add_node("mock_interview", run_interview_tool)
    graph.add_node("placement_notes", run_notes_rag_tool)
    graph.add_node("placement_stats", run_stats_tool)
    graph.add_node("format_response", format_response)
    graph.add_node("conversation_context", run_context_tool)
    graph.add_node("general_chat", run_general_chat_tool)
    graph.add_node("compound_query", run_compound_tool)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges("classify_intent", route_by_intent, {
        "resume_score": "resume_score",
        "mock_interview": "mock_interview",
        "placement_notes": "placement_notes",
        "placement_stats": "placement_stats",
        "conversation_context": "conversation_context",
        "general_chat": "general_chat", 
         "compound_query": "compound_query",
    })
    graph.add_edge("resume_score", "format_response")
    graph.add_edge("mock_interview", "format_response")
    graph.add_edge("placement_notes", "format_response")
    graph.add_edge("placement_stats", "format_response")
    graph.add_edge("conversation_context", "format_response") 
    graph.add_edge("general_chat", "format_response")  
    graph.add_edge("compound_query", "format_response")
    graph.add_edge("format_response", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    agent = build_agent()
    config = {"configurable": {"thread_id": "test-session-1"}}

    r1 = agent.invoke({"user_message": "give me a mock interview question on DSA"}, config=config)
    print("Turn 1 intent:", r1["intent"])
    print("History length after turn 1:", len(r1["history"]), "(expect 2)")

    r2 = agent.invoke({"user_message": "give me a harder one"}, config=config)
    print("\nTurn 2 intent:", r2["intent"])
    print("History length after turn 2:", len(r2["history"]), "(expect 4)")