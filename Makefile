SHELL := /bin/bash
.PHONY: help test test-cov lint format typecheck clean install install-dev ci pre-commit

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

install-dev: install ## Install dev dependencies
	pip install -e ".[dev]"

test: ## Run unit tests (skip integration/online)
	python -m pytest tests/ -v --ignore=tests/test_orbit_utils.py \
		-m "not integration and not online" \
		--tb=short

test-all: ## Run all tests (including slow)
	python -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage
	python -m pytest tests/ -v --tb=short \
		--ignore=tests/test_orbit_utils.py \
		-m "not integration and not online" \
		--cov=krpc_rendezvous \
		--cov-report=term-missing \
		--cov-report=html
	@echo "Coverage report: open htmlcov/index.html"

lint: ## Lint with ruff
	python -m ruff check krpc_rendezvous/ tests/

format: ## Format with ruff
	python -m ruff format krpc_rendezvous/ tests/

format-check: ## Check formatting without changes
	python -m ruff format --check krpc_rendezvous/ tests/

typecheck: ## Type-check with mypy
	python -m mypy krpc_rendezvous/ tests/

clean: ## Clean build/test artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage/
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete

pre-commit: lint format-check typecheck test ## Run all pre-commit checks

ci: lint format-check typecheck test ## CI pipeline (GitHub Actions)

.DEFAULT_GOAL := help
