.PHONY: test eval run lint format typecheck clean help

PYTHON ?= python3
AKARI_DATA_DIR ?= $(HOME)/.cache/akari

help:
	@echo "Akari CLI & Testing Utility"
	@echo "----------------------------"
	@echo "make test      - Run deterministic offline test suite"
	@echo "make eval      - Run retrieval & grounding evaluation benchmark"
	@echo "make lint      - Lint codebase with Ruff"
	@echo "make format    - Format codebase with Ruff"
	@echo "make typecheck - Run Mypy static type checking"
	@echo "make run       - Run CLI with default prompt"
	@echo "make clean     - Clean cached sessions, temporary files, and test caches"

test:
	AKARI_OFFLINE_TEST=1 $(PYTHON) -m pytest tests/ -v

eval:
	$(PYTHON) -m agent.evaluation

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

typecheck:
	$(PYTHON) -m mypy agent/ tests/

run:
	$(PYTHON) main.py

clean:
	rm -rf __pycache__ agent/__pycache__ agent/nodes/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf /tmp/akari_pdfs /tmp/akari_chroma
	rm -rf $(AKARI_DATA_DIR)/chroma $(AKARI_DATA_DIR)/pdfs
