SHELL := /bin/bash
.SHELLFLAGS := -e -c

.PHONY: sync
sync:
	uv sync --all-extras --all-packages --group dev

.PHONY: format
format: 
	uv run ruff format
	uv run ruff check --fix

.PHONY: format-check
format-check:
	uv run ruff format --check

.PHONY: lint
lint: 
	uv run ruff check

.PHONY: build-ui
build-ui:
	uv pip install build
	npm --version || echo "npm not found, please install npm"
	cd ebs/runtime/ui/frontend && npm install && bash build.sh
	uv pip install --force-reinstall ebs/runtime/ui/frontend/build/ebs_agent_ui-0.2.0-py3-none-any.whl
