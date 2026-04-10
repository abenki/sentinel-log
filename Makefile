.PHONY: up down lint test train generate-dataset

N_LOGS ?= 60000
ANOMALY_RATE ?= 0.05
OUTPUT ?= ../../data/logs.jsonl
SEED ?= 42

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

generate-dataset:
	cd services/ingestion && uv run python generate_dataset.py \
		--n-logs $(N_LOGS) \
		--anomaly-rate $(ANOMALY_RATE) \
		--output $(OUTPUT) \
		--seed $(SEED)
