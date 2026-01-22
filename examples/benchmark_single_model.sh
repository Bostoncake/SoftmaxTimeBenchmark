#!/bin/bash
# Example: Benchmark a single model

MODEL=${1:-llama2-7b}

echo "Benchmarking $MODEL..."

python run_benchmark.py \
    --models $MODEL \
    --context-length 8192 \
    --num-tokens 1024 \
    --num-runs 1 \
    --device cuda \
    --dtype float16 \
    --output-dir results

echo "Done! Results saved to results/"
