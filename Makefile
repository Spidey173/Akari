.PHONY: test run lint clean help

help:
	@echo "Akari CLI & Testing Utility"
	@echo "----------------------------"
	@echo "make test     - Run offline test suite"
	@echo "make run      - Run CLI with default prompt"
	@echo "make clean    - Clean cached sessions and temp files"

test:
	python3 -m unittest discover tests

run:
	python3 main.py

clean:
	rm -rf __pycache__ agent/__pycache__ agent/nodes/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	rm -rf /tmp/akari_pdfs /tmp/akari_chroma
