#!/bin/bash
uv run python -m vllm.entrypoints.openai.run_batch -i openai_example_batch.jsonl -o results.jsonl \
    --model Qwen/Qwen3-0.6B \
    --enforce_eager \
    --gpu-memory-utilization 0.6 \
    --max_model_len 1000
    # --max-num-seqs 1 