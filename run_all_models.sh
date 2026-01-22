#!/bin/bash
# Script to run benchmarks on all models with full parameters

echo "Running comprehensive benchmark on all models..."
echo "This will take significant time and requires:"
echo "  - Access to all model weights on Hugging Face"
echo "  - At least 24GB GPU memory"
echo "  - Several hours to complete"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

python run_benchmark.py \
    --models all \
    --context-length 8192 \
    --num-tokens 1024 \
    --num-runs 3 \
    --device cuda \
    --dtype float16 \
    --output-dir results

echo ""
echo "Benchmark complete! Check results/ for output."
