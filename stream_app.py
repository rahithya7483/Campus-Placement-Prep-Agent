"""
streamlit_app.py

Chat UI for the Campus Placement Prep Agent.
Calls the LangGraph agent directly (no FastAPI needed), so it can run on
Streamlit Community Cloud as a single app.

Local run:  streamlit run streamlit_app.py
"""

import os
import sys
import time
import uuid

import streamlit as st

# make src/ importable (agent.py lives there)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from agent import build_agent

st.set_page_config(page_title="Campus Placement Prep Agent", page_icon="\U0001F393")


@st.cache_resource
def get_agent():
    # built once and reused, so MemorySaver keeps the session memory
    return build_agent()


def new_session_id() -> str:
    return str(uuid.uuid4())


def init_state():
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
    if "active_session_id" not in st.session_state:
        sid = new_session_id()
        st.session_state.sessions[sid] = {"label": "New chat", "messages": []}
        st.session_state.active_session_id = sid


def current_messages() -> list:
    return st.session_state.sessions[st.session_state.active_session_id]["messages"]


def maybe_set_label(sid: str, first_message: str):
    sess = st.session_state.sessions[sid]
    if sess["label"] == "New chat":
        sess["label"] = (first_message[:40] + "...") if len(first_message) > 40 else first_message


def stream_words(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)


def send_and_render(text: str):
    sid = st.session_state.active_session_id
    messages = current_messages()

    messages.append({"role": "user", "content": text, "intent": None})
    maybe_set_label(sid, text)
    with st.chat_message("user"):
        st.markdown(text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                agent = get_agent()
                config = {"configurable": {"thread_id": sid}}
                result = agent.invoke({"user_message": text}, config=config)
                answer = result["final_response"]
                intent = result.get("intent")
            except Exception as e:
                answer = f"\u26A0\uFE0F Something went wrong.\n\nDetails: {e}"
                intent = None

        st.write_stream(stream_words(answer))
        if intent:
            st.caption(f"intent: `{intent}`")

    messages.append({"role": "assistant", "content": answer, "intent": intent})


def render_sidebar():
    with st.sidebar:
        st.subheader("Sessions")

        if st.button("\u2795 New session", use_container_width=True):
            sid = new_session_id()
            st.session_state.sessions[sid] = {"label": "New chat", "messages": []}
            st.session_state.active_session_id = sid
            st.rerun()

        for sid in reversed(list(st.session_state.sessions.keys())):
            sess = st.session_state.sessions[sid]
            is_active = sid == st.session_state.active_session_id
            button_label = ("\U0001F7E2 " if is_active else "\u26AA ") + sess["label"]
            if st.button(button_label, key=f"switch_{sid}", use_container_width=True, disabled=is_active):
                st.session_state.active_session_id = sid
                st.rerun()

        st.divider()
        st.subheader("Quick actions")

        quick_actions = [
            ("\U0001F9E9 Mock interview: DSA", "Give me a mock interview question on DSA"),
            ("\U0001F3E2 TCS interview process", "What's the interview process for TCS?"),
            ("\U0001F4C8 Highest package (2024-25)", "What was the highest package offered in 2024-25?"),
        ]
        for label, message in quick_actions:
            if st.button(label, use_container_width=True):
                st.session_state["_pending_message"] = message

        st.divider()
        st.caption(
            "For resume scoring, paste your resume text directly into the chat "
            "(optionally followed by a job description)."
        )


def render_chat():
    st.title("\U0001F393 Campus Placement Prep Agent")
    st.caption(
        "Ask about past placement drives, get a mock interview question, "
        "or paste your resume to get an ATS score."
    )

    for msg in current_messages():
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("intent"):
                st.caption(f"intent: `{msg['intent']}`")

    pending = st.session_state.pop("_pending_message", None)
    user_input = st.chat_input(
        "Ask about placements, request a mock interview question, or paste your resume..."
    )

    to_send = pending or user_input
    if to_send:
        send_and_render(to_send)


def main():
    init_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
