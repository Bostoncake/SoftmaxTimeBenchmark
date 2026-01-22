#!/bin/bash
# Example: Compare LLaMA-2 vs Mistral

echo "Comparing LLaMA-2 7B and Mistral 7B..."

python run_benchmark.py \
    --models llama2-7b mistral-7b \
    --context-length 4096 \
    --num-tokens 512 \
    --num-runs 2 \
    --device cuda \
    --dtype float16 \
    --output-dir results

echo ""
echo "Analyzing results..."
python analyze_results.py results/combined_results_*.json

echo "Done!"
