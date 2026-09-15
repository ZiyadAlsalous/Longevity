.PHONY: help setup check lint format typecheck run

PYTHON ?= python
MIVOLO := git+https://github.com/WildChlamydia/MiVOLO.git@37475e3f8818b5f22448003feec3e64b01bfb188

help:  ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Create the environment and install dependencies (uv preferred)
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --extra dev --extra anthropic --extra vision && \
		uv pip install "setuptools<81" && \
		uv pip install --no-deps --no-build-isolation "$(MIVOLO)"; \
	else \
		$(PYTHON) -m venv .venv && \
		.venv/bin/pip install --upgrade pip "setuptools<81" && \
		.venv/bin/pip install -e ".[dev,anthropic,vision]" && \
		.venv/bin/pip install --no-deps --no-build-isolation "$(MIVOLO)"; \
	fi
	@cp -n .env.example .env || true
	@echo "Environment ready. Add your ANTHROPIC_API_KEY (or OPENAI_API_KEY) to .env."

check: lint typecheck  ## Run ruff and strict mypy

lint:  ## Run ruff lint and format checks
	$(PYTHON) -m ruff check src app.py
	$(PYTHON) -m ruff format --check src app.py

format:  ## Apply ruff formatting and import sorting
	$(PYTHON) -m ruff format src app.py
	$(PYTHON) -m ruff check --fix src app.py

typecheck:  ## Run mypy in strict mode
	$(PYTHON) -m mypy src app.py

run:  ## Launch the Streamlit app
	$(PYTHON) -m streamlit run app.py
