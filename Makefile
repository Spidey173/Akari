.PHONY: test eval run lint format format-check typecheck ci clean help

VENV ?= .venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
AKARI_DATA_DIR ?= $(HOME)/.cache/akari

AKARI_OFFLINE_TEST ?= 1

help:
	@echo "Akari CLI & Testing Utility"
	@echo "----------------------------"
	@echo "make test         - Run deterministic offline test suite"
	@echo "make eval         - Run retrieval & grounding evaluation benchmark"
	@echo "make lint         - Lint codebase with Ruff"
	@echo "make format       - Format codebase with Ruff"
	@echo "make format-check - Check formatting with Ruff"
	@echo "make typecheck    - Run Mypy static type checking"
	@echo "make ci           - Run full CI verification suite locally"
	@echo "make run          - Run CLI with default prompt"
	@echo "make clean        - Clean cached sessions, temporary files, and test caches"

test:
	AKARI_OFFLINE_TEST=1 $(PYTHON) -m pytest tests/ -v

eval:
	AKARI_OFFLINE_TEST=$(AKARI_OFFLINE_TEST) $(PYTHON) -m agent.evaluation

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy agent/ tests/ main.py

ci: lint format-check typecheck test eval

run:
	$(PYTHON) main.py

clean:
	rm -rf __pycache__ agent/__pycache__ agent/nodes/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf /tmp/akari_pdfs /tmp/akari_chroma
	rm -rf $(AKARI_DATA_DIR)/chroma $(AKARI_DATA_DIR)/pdfs
