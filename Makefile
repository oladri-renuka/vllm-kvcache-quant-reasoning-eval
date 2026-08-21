.PHONY: setup prepare run-all run-baseline run-fp8 run-int8 score sanity-check report clean

# Configuration
VENV = venv
PYTHON = $(VENV)/bin/python3
NUM_PROBLEMS ?=

help:
	@echo "vLLM KV-Cache Quantization Experiment"
	@echo "======================================"
	@echo ""
	@echo "Available targets:"
	@echo "  make setup           - Install dependencies (run first)"
	@echo "  make prepare         - Prepare test dataset"
	@echo "  make run-all         - Run all three experiments (auto, fp8, int8_per_token_head)"
	@echo "  make run-baseline    - Run with auto (baseline)"
	@echo "  make run-fp8         - Run with fp8"
	@echo "  make run-int8        - Run with int8_per_token_head"
	@echo "  make score           - Score all results"
	@echo "  make sanity-check    - Manual review of samples"
	@echo "  make report          - Show summary report"
	@echo "  make clean           - Remove results and venv"
	@echo ""
	@echo "Variables:"
	@echo "  NUM_PROBLEMS=N       - Limit to N problems (for testing)"
	@echo ""
	@echo "Example usage:"
	@echo "  make setup"
	@echo "  make prepare"
	@echo "  make run-all NUM_PROBLEMS=10  # Test with 10 problems"
	@echo "  make score"
	@echo ""

setup:
	bash scripts/setup.sh

prepare: $(VENV)
	$(PYTHON) scripts/prepare_testset.py

run-all: run-baseline run-fp8 run-int8

run-baseline: $(VENV)
	$(PYTHON) scripts/run_experiment.py --kv_cache_dtype auto $(if $(NUM_PROBLEMS),--num_problems $(NUM_PROBLEMS))

run-fp8: $(VENV)
	$(PYTHON) scripts/run_experiment.py --kv_cache_dtype fp8 $(if $(NUM_PROBLEMS),--num_problems $(NUM_PROBLEMS))

run-int8: $(VENV)
	$(PYTHON) scripts/run_experiment.py --kv_cache_dtype int8_per_token_head $(if $(NUM_PROBLEMS),--num_problems $(NUM_PROBLEMS))

score: $(VENV)
	$(PYTHON) scripts/score_outputs.py

sanity-check: $(VENV)
	$(PYTHON) scripts/sanity_check.py --num_samples 5

report: score sanity-check
	@echo ""
	@echo "Report data available in results/*/{scores.json,outputs.jsonl}"
	@echo ""

clean:
	rm -rf venv results/

$(VENV):
	@echo "Virtual environment not found. Run 'make setup' first."
	@exit 1

.DEFAULT_GOAL := help
