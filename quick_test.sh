#!/bin/bash
# Quick test script to verify the benchmark setup works

echo "Running quick test with reduced parameters..."
echo "This will test with:"
echo "  - Single model: LLaMA-2 7B"
echo "  - Context: 512 tokens (reduced from 8192)"
echo "  - Generate: 128 tokens (reduced from 1024)"
echo ""

python run_benchmark.py \
    --models llama2-7b \
    --context-length 512 \
    --num-tokens 128 \
    --num-runs 1 \
    --output-dir test_results

echo ""
echo "Test complete! Check test_results/ for output."
