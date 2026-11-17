.PHONY: help install test lint numbers figures significance folds reported unfiltered nested all-runs

help:
	@echo "install       install dependencies with uv"
	@echo "test          run the unit tests"
	@echo "lint          run ruff"
	@echo "numbers       recompute every accuracy figure quoted in the paper"
	@echo "significance  recompute the paired tests, shared folds and nested"
	@echo "figures       redraw Figures 2, 3, 4 and S1"
	@echo "folds         rewrite artifacts/folds.csv"
	@echo "reported      refit the reported configuration from the raw data"
	@echo "unfiltered    refit the same pipeline with the filter switched off"
	@echo "nested        refit both nested constructions"
	@echo "all-runs      reported + unfiltered + nested"

install:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check .

numbers:
	uv run python run/pipeline/paper_numbers.py

significance:
	uv run python run/pipeline/significance_tests.py
	uv run python run/pipeline/significance_nested.py

figures:
	uv run python run/pipeline/make_figures.py

folds:
	uv run python run/pipeline/export_folds.py

reported:
	uv run tbm-rop all kalman.mode=causal kalman.process_noise=0.1 \
		paths.processed=data/processed/sig_a paths.outputs=outputs/sig_a

unfiltered:
	uv run tbm-rop all kalman.mode=none \
		paths.processed=data/processed/sig_a_off paths.outputs=outputs/sig_a_off

nested:
	uv run tbm-rop nested kalman.mode=causal \
		paths.processed=data/processed/a paths.outputs=outputs/nested_a
	uv run tbm-rop nested kalman.mode=none \
		paths.processed=data/processed/a_off paths.outputs=outputs/nested_a_off

all-runs: reported unfiltered nested
