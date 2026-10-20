.PHONY: install data train test lint notebooks report serve docker compose deploy-kind clean-kind all
install:
	python -m pip install -r requirements.txt
	python -m pip install -e . --no-deps
data:
	python scripts/download_data.py
train:
	python -m heartops.train
test:
	python -m pytest -q
lint:
	python -m ruff check src tests scripts
notebooks:
	python scripts/build_notebooks.py
report:
	python scripts/architecture_diagram.py
	python scripts/build_report.py
serve:
	uvicorn heartops.api:app --host 0.0.0.0 --port 8000
docker:
	docker build -t heartops:local .
compose:
	docker compose up -d --build
deploy-kind:
	./scripts/deploy_kind.sh
clean-kind:
	kind delete cluster --name heartops
all: lint train test notebooks report docker
