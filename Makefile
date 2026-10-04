# ──────────────────────────────────────────────────────────────────────────────
# VajraDrishti Makefile
#
# Targets:
#   make setup          — create venv and install all requirements
#   make synthetic      — generate synthetic storm data
#   make sevir-download — download a small SEVIR subset from S3
#   make train          — run train.py with default config
#   make evaluate       — run evaluate.py, save metrics JSON + plots
#   make demo           — generate synthetic data, run a quick train, start API
#   make api            — start the FastAPI backend
#   make test           — run all pytest tests
#   make lint           — run ruff linter
#   make clean          — remove venv, checkpoints, results
# ──────────────────────────────────────────────────────────────────────────────

PYTHON     := python3.11
VENV       := venv
PIP        := $(VENV)/bin/pip
PY         := $(VENV)/bin/python
RUFF       := $(VENV)/bin/ruff
PYTEST     := $(VENV)/bin/pytest
UVICORN    := $(VENV)/bin/uvicorn
CONFIG     := configs/config.yaml

.PHONY: setup synthetic sevir-download train evaluate demo api test lint clean help

# Default target — show help
help:
	@echo ""
	@echo "  VajraDrishti — available make targets"
	@echo "  ──────────────────────────────────────"
	@echo "  make setup          Create venv + install requirements"
	@echo "  make synthetic      Generate synthetic storm data"
	@echo "  make sevir-download Download small SEVIR subset from S3"
	@echo "  make train          Train the U-Net + ConvLSTM model"
	@echo "  make evaluate       Evaluate model and baseline, save metrics"
	@echo "  make demo           End-to-end demo (synthetic → train → API)"
	@echo "  make api            Start FastAPI backend"
	@echo "  make test           Run pytest test suite"
	@echo "  make lint           Run ruff linter"
	@echo "  make clean          Remove venv, checkpoints, results"
	@echo ""

# ── Environment setup ─────────────────────────────────────────────────────────
setup:
	@echo "==> Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "==> Upgrading pip..."
	$(PIP) install --upgrade pip
	@echo "==> Installing requirements..."
	$(PIP) install -r requirements.txt
	@echo "==> Creating required directories..."
	mkdir -p data checkpoints results docs/images notebooks
	touch data/.gitkeep
	@echo "==> Setup complete. Activate with: source venv/bin/activate"

# ── Data generation ───────────────────────────────────────────────────────────
synthetic:
	@echo "==> Generating synthetic storm data..."
	$(PY) scripts/make_synthetic.py --config $(CONFIG)
	@echo "==> Synthetic data written to data/synthetic/"

sevir-download:
	@echo "==> Downloading small SEVIR subset..."
	$(PY) scripts/download_sevir_subset.py --config $(CONFIG)
	@echo "==> SEVIR subset written to data/sevir/"

# ── Training ──────────────────────────────────────────────────────────────────
train:
	@echo "==> Starting training..."
	$(PY) -m src.train --config $(CONFIG)

# ── Evaluation ────────────────────────────────────────────────────────────────
evaluate:
	@echo "==> Running evaluation..."
	$(PY) -m src.evaluate --config $(CONFIG)
	@echo "==> Results saved to results/"

# ── Demo (used by fresh-clone instructions in README) ─────────────────────────
demo: synthetic
	@echo "==> Running short training (5 epochs) for demo..."
	$(PY) -m src.train --config $(CONFIG) --max-epochs 5
	@echo "==> Evaluating..."
	$(PY) -m src.evaluate --config $(CONFIG)
	@echo "==> Starting API in background (PID saved to .api.pid)..."
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 &
	echo $$! > .api.pid
	@echo ""
	@echo "==> Demo running!"
	@echo "    API & Dashboard: http://localhost:8000"
	@echo "    API Docs:        http://localhost:8000/docs"
	@echo "    Stop API:        kill \$$(cat .api.pid)"

# ── API server ────────────────────────────────────────────────────────────────
api:
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --reload

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v

# ── Linting ───────────────────────────────────────────────────────────────────
lint:
	$(RUFF) check src/ api/ scripts/ tests/
	$(RUFF) format --check src/ api/ scripts/ tests/

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	@echo "==> Removing venv..."
	rm -rf $(VENV)
	@echo "==> Removing checkpoints..."
	rm -rf checkpoints/
	@echo "==> Removing results..."
	rm -rf results/
	@echo "==> Done."
