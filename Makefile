.PHONY: setup test lint run_api run_eval run_all

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest

lint:
	python -m compileall src scripts evaluation tests

run_api:
	python scripts/run_api.py --config configs/dev.yaml

run_eval:
	python scripts/run_eval.py --config configs/dev.yaml

run_all:
	python scripts/run_all.py --config configs/dev.yaml
