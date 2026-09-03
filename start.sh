#!/bin/bash
set -e

if [ ! -f data/processed/placement.db ]; then
    echo "Building placement.db..."
    python build_db.py
fi

cd src
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

sleep 5

# Render assigns its own port via $PORT - default to 7860 if not set
# (e.g. when running locally with `docker run`)
STREAMLIT_PORT="${PORT:-7860}"

streamlit run streamlit_app.py \
    --server.port "$STREAMLIT_PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false

kill $BACKEND_PID 2>/dev/null || true