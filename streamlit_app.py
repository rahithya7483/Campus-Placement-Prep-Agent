"""
streamlit_app.py

Chat UI for the Campus Placement Prep Agent. Talks to the existing FastAPI
backend (src/main.py) over HTTP.

Run:
    1. In one terminal: cd src && uvicorn main:app --reload
    2. In another terminal (project root): streamlit run streamlit_app.py

Note on sessions: the backend (LangGraph's MemorySaver) already keeps full
conversation state per session_id for as long as the uvicorn process is
alive - switching back to an old session_id here correctly resumes it on
the backend. This file just needs to remember, on the frontend, which
session_ids exist and what their chat bubbles looked like, so you can
switch between them in the sidebar. This history is local to this browser
tab and this Streamlit run - it does not survive a full app restart.
"""

import time
import uuid

import requests
import streamlit as st

API_URL = "http://localhost:8000/chat"
HEALTH_URL = "http://localhost:8000/health"

st.set_page_config(page_title="Campus Placement Prep Agent", page_icon="\U0001F393")


def new_session_id() -> str:
    return str(uuid.uuid4())


def init_state():
    if "sessions" not in st.session_state:
        # session_id -> {"label": str, "messages": [{"role","content","intent"}]}
        st.session_state.sessions = {}
    if "active_session_id" not in st.session_state:
        sid = new_session_id()
        st.session_state.sessions[sid] = {"label": "New chat", "messages": []}
        st.session_state.active_session_id = sid


def current_messages() -> list:
    return st.session_state.sessions[st.session_state.active_session_id]["messages"]


def backend_is_up() -> bool:
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def maybe_set_label(sid: str, first_message: str):
    sess = st.session_state.sessions[sid]
    if sess["label"] == "New chat":
        sess["label"] = (first_message[:40] + "...") if len(first_message) > 40 else first_message


def stream_words(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)


def send_and_render(text: str):
    """Appends the user message, calls the backend, and renders the reply
    with a word-by-word reveal - all within the current script run, no
    st.rerun() needed for the reply itself to show up correctly."""
    sid = st.session_state.active_session_id
    messages = current_messages()

    messages.append({"role": "user", "content": text, "intent": None})
    maybe_set_label(sid, text)
    with st.chat_message("user"):
        st.markdown(text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    API_URL,
                    json={"message": text, "session_id": sid},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["response"]
                intent = data.get("intent")
            except requests.exceptions.RequestException as e:
                answer = f"\u26A0\uFE0F Couldn't reach the backend at `{API_URL}`.\n\nDetails: {e}"
                intent = None

        st.write_stream(stream_words(answer))
        if intent:
            st.caption(f"intent: `{intent}`")

    messages.append({"role": "assistant", "content": answer, "intent": intent})


def render_sidebar():
    with st.sidebar:
        st.subheader("Backend")
        if backend_is_up():
            st.success("Connected", icon="\u2705")
        else:
            st.error(
                "Not reachable - start it with:\n\ncd src && uvicorn main:app --reload",
                icon="\u26A0\uFE0F",
            )

        st.divider()
        st.subheader("Sessions")

        if st.button("\u2795 New session", use_container_width=True):
            sid = new_session_id()
            st.session_state.sessions[sid] = {"label": "New chat", "messages": []}
            st.session_state.active_session_id = sid
            st.rerun()

        # most recently created sessions first
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
            "(optionally followed by a job description) - the agent reads it "
            "straight from the message."
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