#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
backend_pid=""
frontend_pid=""

cleanup() {
  trap - EXIT INT TERM
  [[ -z "$backend_pid" ]] || kill "$backend_pid" 2>/dev/null || true
  [[ -z "$frontend_pid" ]] || kill "$frontend_pid" 2>/dev/null || true
  wait "$backend_pid" "$frontend_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

backend_python="$project_dir/backend/.venv/bin/python"
if [[ -x "$project_dir/backend/.venv-ml/bin/python" ]]; then
  backend_python="$project_dir/backend/.venv-ml/bin/python"
fi

if [[ ! -x "$backend_python" ]]; then
  echo "Backend is not installed. Run: make setup"
  exit 1
fi

if [[ ! -d "$project_dir/frontend/node_modules" ]]; then
  echo "Frontend is not installed. Run: make setup"
  exit 1
fi

cd "$project_dir"
"$backend_python" -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 &
backend_pid=$!

(
  cd frontend
  npm run dev
) &
frontend_pid=$!

echo "Backend: http://localhost:8000/docs"
echo "Frontend: http://localhost:3000"
echo "Backend runtime: $($backend_python --version 2>&1)"
echo "Press Ctrl+C to stop both services."

while kill -0 "$backend_pid" 2>/dev/null && kill -0 "$frontend_pid" 2>/dev/null; do
  sleep 1
done

echo "A service stopped; shutting down the application."
exit 1
