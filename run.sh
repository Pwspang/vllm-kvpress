#!/bin/bash
python -m vllm.entrypoints.openai.run_batch -i openai_example_batch.jsonl -o results.jsonl \
    --model /home/users/ntu/wpang010/scratch/models/DeepSeek-R1-0528-Qwen3-8B \
    --enforce_eager \
    --gpu-memory-utilization 0.6 \
    --max_model_len 1000
    # --max-num-seqs 1 
