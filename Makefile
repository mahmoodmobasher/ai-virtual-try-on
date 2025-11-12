PYTHON ?= python3
BACKEND_DIR := backend
FRONTEND_DIR := frontend
BACKEND_VENV := $(BACKEND_DIR)/.venv
ACTIVATE := . $(BACKEND_VENV)/bin/activate

.PHONY: setup setup-backend setup-frontend dev dev-backend dev-frontend fmt lint migrate

setup: setup-backend setup-frontend

setup-backend:
	@echo ">> Setting up backend virtual environment"
	@test -d $(BACKEND_VENV) || $(PYTHON) -m venv $(BACKEND_VENV)
	@$(ACTIVATE) && pip install --upgrade pip wheel
	@$(ACTIVATE) && pip install -r $(BACKEND_DIR)/requirements.txt

setup-frontend:
	@echo ">> Installing frontend dependencies"
	@cd $(FRONTEND_DIR) && npm install

dev:
	@echo "Run 'make dev-backend' and 'make dev-frontend' in separate terminals."

dev-backend:
	@$(ACTIVATE) && uvicorn app.main:app --reload --factory --port 8000 --app-dir $(BACKEND_DIR)/app

dev-frontend:
	@cd $(FRONTEND_DIR) && npm run dev

fmt:
	@$(ACTIVATE) && ruff format $(BACKEND_DIR)/app

lint:
	@$(ACTIVATE) && ruff check $(BACKEND_DIR)/app

migrate:
	@$(ACTIVATE) && alembic upgrade head

