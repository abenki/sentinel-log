.PHONY: up down lint test train

up:
	docker compose up --build -d

down:
	docker compose down -v

lint:
	cd services/ingestion && uv run ruff check .
	cd services/ml && uv run ruff check .
	cd services/api && uv run ruff check .

test:
	cd services/ingestion && uv run pytest
	cd services/ml && uv run pytest
	cd services/api && uv run pytest

train:
	cd services/ml && uv run python train.py
