# rxndata pipeline. `make all` runs the whole thing; per-phase targets too.
# Uses the repo-local venv at .venv (created by `make setup`).

PY := .venv/bin/python
PIP := .venv/bin/pip
export PYTHONPATH := src

.PHONY: all setup deps clone-omebench ontology phase1 test lint clean

all: phase1        ## (phases 2-9 appended as they land)

setup: .venv deps clone-omebench ## create venv, install deps, clone oMeBench

.venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip

deps: ## install pinned deps (base + ord/atommap extras as needed)
	$(PIP) install rdkit==2025.9.2 pyyaml==6.0.2 pandas==2.2.3 polars==1.36.1 \
		pyarrow==21.0.0 numpy==1.26.4 requests==2.32.3 tqdm==4.67.1 \
		py2opsin==1.2.0 py7zr==1.0.0 pytest==8.3.5 ruff==0.9.10
	# ord-schema protobufs, without its heavy psycopg2/flask deps:
	$(PIP) install --no-deps ord-schema==0.3.37 "protobuf>=3.20,<5"

clone-omebench: ## vendor the official oMeBench repo for the reference scorer
	@test -d third_party/oMeBench || \
		git clone --depth 1 https://github.com/skylarkie/oMeBench.git third_party/oMeBench

ontology: ## (re)generate configs/ontology.json from the benchmark artifacts
	$(PY) -m rxndata.ontology

phase1: ontology ## Phase 1: ingest all enabled + license-cleared sources
	$(PY) -m rxndata.phase1_ingest --json-out data/interim/_phase1_gate.json

test: ## run the pytest suite (schema, validators, ingest smoke)
	$(PY) -m pytest tests/ -q

lint:
	.venv/bin/ruff check src tests scripts

clean: ## remove interim outputs (keeps raw cache + final)
	rm -rf data/interim/*.parquet data/interim/_*.json
