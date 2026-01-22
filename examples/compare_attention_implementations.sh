#!/bin/bash
# Compare different attention implementations for the same model

echo "Comparing attention implementations for LLaMA-2 7B"
echo "This will run the same benchmark with eager, SDPA, and Flash Attention 2"
echo ""

MODEL="llama2-7b"
CONTEXT=2048
TOKENS=256

echo "1. Eager Attention (exact softmax measurement, slowest)..."
python run_benchmark.py \
    --models $MODEL \
    --context-length $CONTEXT \
    --num-tokens $TOKENS \
    --attn-implementation eager \
    --output-dir results/attention_comparison

echo ""
echo "2. SDPA (estimated softmax, faster)..."
python run_benchmark.py \
    --models $MODEL \
    --context-length $CONTEXT \
    --num-tokens $TOKENS \
    --attn-implementation sdpa \
    --output-dir results/attention_comparison

echo ""
echo "3. Flash Attention 2 (no softmax breakdown, fastest)..."
echo "   Note: This requires flash-attn package installed"
python run_benchmark.py \
    --models $MODEL \
    --context-length $CONTEXT \
    --num-tokens $TOKENS \
    --attn-implementation flash_attention_2 \
    --output-dir results/attention_comparison

echo ""
echo "Done! Check results/attention_comparison/ for outputs"
echo ""
echo "Expected results:"
echo "  - Eager: Slowest, exact softmax measurement"
echo "  - SDPA: ~30% faster, estimated softmax (~10% of SDPA time)"
echo "  - Flash2: ~60% faster, no softmax breakdown available"
