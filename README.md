# Campus Placement Prep Agent

An AI assistant for students preparing for campus placements. It answers questions about
past placement drives, scores resumes against ATS (Applicant Tracking System) standards,
generates mock interview questions, and recalls company-specific interview processes from
historical data.

## What it can do

- **Placement statistics** — numeric questions like "highest package in 2024-25" or
  "how many companies visited last year," answered by querying real historical data.
- **Company interview notes** — qualitative questions like "what's the interview process
  for TCS?" or "what technical questions does Cognizant ask?"
- **Mock interview practice** — generates a new practice question on a topic/role you pick.
- **Resume ATS scoring** — paste your resume (with or without a job description) and get a
  score, missing keywords, formatting issues, and top fixes.
- **Conversation memory** — remembers what you've discussed earlier in the session, so you
  can ask follow-ups like "is this person able to get hired?" after a resume was scored.
- **Compound questions** — can chain a stats lookup into a notes lookup in one go, e.g.
  "what's the interview process of the company that paid the highest package in 2024-25?"

## Architecture

The project is a single LangGraph agent sitting behind a FastAPI backend, with a Streamlit
chat UI on top.

```
Streamlit UI  --HTTP-->  FastAPI (/chat)  -->  LangGraph agent
                                                    |
                                    +---------------+---------------+
                                    |               |               |
                              classify_intent   (routes to one or more tools below)
                                    |
        +----------------+----------------+----------------+------------------+
        |                |                |                |                  |
   resume_score     mock_interview   placement_notes   placement_stats   conversation_context
   (ATS scoring)    (practice Qs)    (RAG / Chroma)    (text-to-SQL)     (answers from history)
```

**Two data stores, deliberately split by the kind of question they answer:**
- **SQLite (`placement.db`)** — structured numeric data (`placement_stats` table). Used for
  anything requiring counting, comparing, or aggregating across many rows — things a vector
  search structurally can't do well.
- **ChromaDB (vector store)** — qualitative company notes (interview rounds, sample
  questions, CGPA cutoffs), chunked *per field* per company, so a question about "technical
  questions" doesn't drag back an entire company's unrelated HR-round text too.

**Session memory** is handled by LangGraph's `MemorySaver` checkpointer, keyed by
`session_id`. As long as the backend process stays running, a session's full conversation
history is available for follow-up questions and summaries — even if you switch away and
back to it in the UI.

## Tech stack

- **Backend**: FastAPI, LangGraph (agent orchestration), Groq (LLM inference,
  `openai/gpt-oss-20b`)
- **Data**: SQLite (stats), ChromaDB + `sentence-transformers` (`all-MiniLM-L6-v2`) for
  semantic search over company notes
- **UI**: Streamlit
- **Validation**: Pydantic (structured outputs for resume scores and interview questions)

## Project structure

```
campus_placement_prep_agent/
├── data/
│   ├── raw/
│   │   ├── college_placement_statistics_2016_17_to_2025_26.csv
│   │   └── company_notes_placement_agent.csv
│   └── processed/
│       ├── chroma_db/            (vector store, built by the notebook)
│       ├── placement.db          (SQLite, built by build_db.py)
│       └── placement_cleaned.csv
├── notebooks/
│   └── 01_rag_pipeline.ipynb     (builds the Chroma vector store)
├── src/
│   ├── tools/
│   │   ├── mock_interview.py
│   │   ├── placement_rag.py
│   │   ├── placement_stats.py
│   │   └── resume_scorer.py
│   ├── agent.py                  (LangGraph orchestration)
│   ├── llm_client.py             (Groq client, with a mock mode for offline dev)
│   └── main.py                   (FastAPI app)
├── streamlit_app.py               (chat UI)
├── build_db.py                    (raw CSVs -> SQLite)
├── dedupe_companies.py            (one-time data-cleaning pass, see below)
├── Dockerfile                     (single container, see Deployment section)
├── start.sh                       (starts backend + UI inside the container)
├── .dockerignore
├── .gitignore
├── requirements.txt
└── .env                           (GROQ_API, MOCK_LLM - never committed)
```

## Setup & running

```bash
pip install -r requirements.txt
python build_db.py                 # builds placement.db from the raw CSVs
# run notebooks/01_rag_pipeline.ipynb once to build the Chroma vector store
```

Two terminals:
```bash
# Terminal 1 - backend
cd src
uvicorn main:app --reload

# Terminal 2 - UI
streamlit run streamlit_app.py
```

Set `MOCK_LLM=true` in `.env` to develop/test without hitting the real Groq API (it returns
canned responses shaped like real ones). Set it to `false` with a valid `GROQ_API` key for
real answers.

## Deployment

The app is containerized with Docker and deployed on **Render.com** (free tier). Both the
FastAPI backend and the Streamlit UI run inside a **single container** — Render's free tier
exposes only one port per service, so `start.sh` starts `uvicorn` in the background on an
internal port (8000) and runs Streamlit in the foreground on the port Render assigns
(`$PORT`), with the UI talking to the backend over `localhost` since they share the same
container.

**Why Render, not Hugging Face Spaces**: HF Spaces was the original plan, but as of this
writing, creating a Docker (or Gradio) Space requires a paid PRO plan — only fully static
Spaces are free. On top of that, HF's billing doesn't yet support Indian-issued credit
cards. Render's free tier needed no card at all and accepts a plain Dockerfile deploy, so
that's what this project actually runs on. Worth rechecking Hugging Face's pricing page
before assuming this is still accurate if you're reading this much later — platform pricing
policies change.

**What to expect from the deployed app** — none of this is a bug, it's normal free-tier
container behavior:
- **Cold starts.** After roughly 15 minutes with no traffic, Render spins the container down
  completely. The next visit takes 30-50 seconds to wake back up before the UI responds.
- **First-message delay.** The very first chat message after a cold start is slower than
  normal LLM latency alone would suggest, because `sentence-transformers` has to download
  and load the `all-MiniLM-L6-v2` embedding model into memory on first use.
- **Conversation memory resets on every spin-down/restart.** `MemorySaver` is in-process —
  it does not survive the container restarting, which happens on every cold start and every
  redeploy. This is the same limitation noted below for local runs, just more frequent here
  because free-tier containers restart far more often than a laptop does.
- **`start.sh` waits for the backend before starting the UI**, polling `/health` for up to 60
  seconds rather than assuming a fixed delay is enough — an early version used a blind
  `sleep 5`, which occasionally raced ahead of a slow cold-start backend and briefly showed
  a "couldn't reach the backend" error in the UI.

**Live LINK** -- `https://campus-placement-prep-agent.onrender.com/`

**To redeploy after a code change**: push to GitHub (`git push origin main`) — if
Auto-Deploy is enabled on the Render service (Settings tab), a new build starts
automatically. Otherwise trigger it manually from the service page (Manual Deploy → Deploy
latest commit).

**Environment variables/secrets**, set in Render's dashboard under the service's
Environment tab (never committed to the repo):
- `GROQ_API` — your Groq API key
- `MOCK_LLM` — `false` for the deployed app to use the real model
- `PORT` — `10000` (Render's expected default; `start.sh` falls back to `7860` if unset,
  which matters if you ever run the same image locally with `docker run`)


## The development journey: problems faced, and how they were fixed

This project went through a real back-and-forth debugging process. Below is a few records of
what actually went wrong along the way and how each issue was resolved — kept here.

### 1. RAG chunks were too coarse
**Problem**: the first version of the vector-search pipeline embedded one giant chunk per
company, mixing interview process, technical questions, HR questions, and CGPA cutoff into
a single blob. Asking about just "technical questions" would drag back the entire company's
unrelated text too, diluting retrieval quality.
**Fix**: split into separate chunks per field (`interview_process`, `role_type`,
`cgpa_cutoff`, `technical_questions`, `hr_questions`), each tagged with metadata so the
search can optionally filter by field. Also removed the numeric stats data from the vector
store entirely — once the SQL tool handled all numeric questions, having those chunks in
Chroma was just noise that could dilute the notes search.

### 2. Real LLM output sometimes wasn't valid JSON
**Problem**: the mock LLM never wrapped its output in markdown code fences, so this was
invisible during early testing — but the real Groq model sometimes does
(` ```json ... ``` `), which broke the Pydantic parsing for resume scores and interview
questions.
**Fix**: added fence-stripping before parsing, consistent with what the SQL-generation tool
was already doing.

### 3. "No data" queries weren't actually empty
**Problem**: SQL aggregate functions like `MAX()` or `COUNT()` always return exactly one row
— even when nothing matches the filter — just with `NULL`/`0` values. The synthesis step's
"if the result is empty, say so" instruction never fired, because the result technically
wasn't an empty list; it was misleading.
**Fix**: added a check that treats an all-`NULL` row the same as a truly empty result,
before ever calling the LLM to "explain" it.

### 4. Resume scoring couldn't handle "just paste my resume, no job description"
**Problem**: the resume-scoring tool always expected a job description; without one, the
classifier was instructed to bail out entirely to the notes-search tool, so a resume pasted
alone got a "no context" response instead of a score.
**Fix**: added a second scoring path — general ATS/resume-quality best practices (action
verbs, quantified impact, standard section headers) — used automatically whenever no job
description is given.

### 5. The agent had no memory for its own conversation
**Problem**: questions like "summarize what I've asked" or "is this person able to get
hired?" (after a resume had already been discussed) fell into the notes-search tool, which
has no concept of the conversation itself — so it always answered "I don't have that
information," even though the answer was sitting right there in the chat history.
**Fix**: added a dedicated intent that answers directly from the full conversation history,
separate from company-notes search.

### 6. Data quality: the same company was recorded under different spellings
**Problem**: the single most impactful bug found. The raw scraped data recorded some
companies under multiple different names — e.g. "TCS" and "Tata Consultancy Services" as
two entirely separate entries, or "BYJU's" / "Byjus" / "BYJU`S" as three. This meant a
question about "TCS" silently missed all the data filed under "Tata Consultancy Services,"
producing confidently wrong or incomplete answers with no visible error.
**Fix**: wrote a standalone deduplication script (`dedupe_companies.py`) that merges
spelling variants in three tiers of confidence — (1) exact match after stripping
case/punctuation, (2) exact match after also stripping legal suffixes like "Pvt Ltd," and
(3) a small hand-verified list of acronym/full-name pairs (TCS, SLS, etc.). Anything that
doesn't clear one of those three bars is written to a review file for manual eyeballing
instead of being auto-merged — a broad fuzzy-matching pass alone produced far too many false
positives (different companies that just happen to share a common word like
"Technologies").

### 7. SQL string matching was too strict on capitalization
**Problem**: even after de-duplication, the AI generating SQL queries would sometimes guess
the wrong capitalization for a company name (e.g. `'Byju's'` instead of the stored
`"BYJU's"`), and SQLite's default string comparison is case-sensitive — so a technically
correct-looking query would still match zero rows.
**Fix**: instructed the SQL-generation step to always compare company names using SQLite's
`COLLATE NOCASE`, which fixes the mismatch regardless of what casing the model guesses.

### 8. Year formats didn't match between user phrasing and stored data
**Problem**: users naturally write "2024-2025," but the data stores years as "2024-25."
An exact-match SQL query against the wrong format returned nothing.
**Fix**: added explicit normalization instructions and examples to the SQL-generation
prompt, so the model converts any year phrasing to the correct stored format before
querying.

### 9. Ambiguous single-year questions didn't disclose their assumption
**Problem**: a question like "highest package in 2025" is genuinely ambiguous — it could
mean the 2024-25 or 2025-26 academic year — and the system would silently pick one without
telling the user which.
**Fix**: added detection for this ambiguity, and the final answer now explicitly states
which academic year it assumed.

### 10. An empty or placeholder resume still got a full, fabricated score
**Problem**: submitting "N/A" as a resume still produced a confident, detailed-looking ATS
score with fabricated missing keywords and feedback — arguably worse than an error, since it
looked completely legitimate.
**Fix**: added input validation that rejects blank, placeholder, or implausibly short text
*before* it ever reaches the LLM, asking the user to paste an actual resume instead.

### 11. Questions needing two tools at once weren't supported
**Problem**: a question like "what's the interview process of the company with the highest
package in 2024-25?" needs a stats lookup *and* a notes lookup chained together — the
original design could only route to one tool per question.
**Fix**: added a compound-query path that runs the stats lookup first, extracts the
resulting company, feeds it into the notes search, and synthesizes one combined answer.

### 12. Greetings and goodbyes were being mishandled
**Problem**: at first, "hi" fell into the notes-search tool and got an unhelpful "the
context doesn't contain an answer." After adding a dedicated small-talk intent, "thank you"
and "bye" both got the *same* long feature-introduction message repeated verbatim — clearly
wrong for someone ending the conversation.
**Fix**: split small talk into two subtypes — an opening greeting (full intro) and a closing
(short, appropriate goodbye) — so the response actually matches what the user is doing.

## Known limitations

- **Interview-notes coverage is thin.** Only 13 of the 550+ companies in the placement
  statistics have any recorded interview process/notes data. Can enlarge details of company at a any time manually. Asking about a company outside that set will either honestly say there's no data, or — since the search has no similarity threshold — could occasionally surface a different, unrelated company's notes as if they were relevant.
- **No user authentication or session isolation.** Anyone who reuses the same `session_id`
  can see and reference another person's conversation, including any resume they pasted.
  Fine for a personal project or classroom demo; not fine for any kind of shared or public
  deployment as-is.
- **Session memory doesn't survive a server restart.** `MemorySaver` is in-process memory —
  restarting the FastAPI server wipes all conversation history.
- **Compound queries only chain one direction.** Stats → notes works (e.g. "the company with
  the highest package, and its interview process"). The reverse — notes → stats (e.g.
  "companies with an HR round, and what they paid") — is not supported.
- **The word-by-word chat "streaming" is not real token streaming.** The full answer is
  computed first, then revealed word by word for effect. Total wait time is unchanged; only
  the perceived experience is improved.
- **Company de-duplication is a manual, one-time step**, not part of the automated data
  pipeline. If the raw source data is ever re-scraped or updated, `dedupe_companies.py`
  needs to be re-run and its review file re-checked by hand.

## Future scope

- **Real token streaming** — convert the Groq call to `stream=True` and expose it through a
  proper Server-Sent-Events endpoint, so answers start appearing as the model generates them
  rather than all at once.
- **Broader interview-notes coverage** — the biggest single quality never left. Expanding
  from 13 to a much larger share of the 550+ companies would make the notes/RAG tool useful
  for far more real questions.
- **A relevance threshold on vector search** — reject or flag retrieved chunks below a
  similarity cutoff, instead of always returning the top-k regardless of how weak the match
  actually is, to reduce the risk of confidently answering with the wrong company's data.
- **Automate the de-duplication pipeline** — fold `dedupe_companies.py`'s logic directly
  into `build_db.py`, so future data refreshes don't require a manual re-run and review step.
- **Bidirectional compound queries** — support the notes → stats chaining direction too.
- **Resume file upload** — accept an actual PDF/DOCX resume upload and extract the text,
  instead of requiring it to be pasted as raw chat text.
- **Persistent, multi-user session storage** — swap `MemorySaver` for a database-backed
  LangGraph checkpointer (SQLite or Postgres), and add basic authentication so sessions are
  actually private per user.
- **Automated test suite** — the manual test cases used throughout development (year
  formats, company-name variants, empty-resume rejection, ambiguous-year disclosure, etc.)
  would make a solid starting point for a real `pytest` regression suite, so future changes
  don't silently reintroduce a fixed bug.