#!/bin/bash
# Example: Run benchmark on CPU (slower, but works without GPU)

echo "Running CPU benchmark (this will be slow)..."

python run_benchmark.py \
    --models llama2-7b \
    --context-length 512 \
    --num-tokens 128 \
    --num-runs 1 \
    --device cpu \
    --dtype float32 \
    --output-dir results

echo "Done!"
