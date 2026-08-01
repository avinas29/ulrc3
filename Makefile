.PHONY: help install dev test lint typecheck bench bench-full ablation perf serve docker clean fmt

PY ?= python3

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

install:  ## install the package with recommended extras
	$(PY) -m pip install ".[tokenizers,fast,server]"

dev:      ## install with dev extras
	$(PY) -m pip install -e ".[all,dev]"

test:     ## run the test suite
	$(PY) -m pytest tests/ -q

lint:     ## ruff
	$(PY) -m ruff check ulrc3 bench tests

fmt:      ## ruff --fix
	$(PY) -m ruff check --fix ulrc3 bench tests

typecheck:  ## mypy
	$(PY) -m mypy ulrc3 --ignore-missing-imports

bench:    ## quick benchmark
	$(PY) -m bench.run_bench --suite all --quick --out bench/results/quick.json

bench-full:  ## full benchmark (all suites, all baselines)
	$(PY) -m bench.run_bench --suite all --out bench/results/full.json

ablation: ## ablation study (falsifies our own claims)
	$(PY) -m bench.ablation --full --mode extreme

perf:     ## throughput and latency profile
	$(PY) -m bench.bench_perf

serve:    ## run the API locally
	$(PY) -m ulrc3.cli serve --port 8000

docker:   ## build the container
	docker build -t ulrc3:1.0.0 .

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} + ; rm -rf .pytest_cache .ruff_cache .mypy_cache
