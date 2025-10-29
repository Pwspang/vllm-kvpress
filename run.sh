#!/bin/bash
uv run python -m vllm.entrypoints.openai.run_batch -i openai_example_batch.jsonl -o results.jsonl \
    --model facebook/opt-125m \
    --enforce_eager \
    --gpu-memory-utilization 0.6 \
    --max-num-seqs 1