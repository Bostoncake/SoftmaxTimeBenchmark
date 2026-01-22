#!/usr/bin/env python3
"""
Main script to run LLM benchmarks for measuring Softmax vs other operation times.
Benchmarks LLaMA-2 7B, LLaMA-3.1 8B, Mistral 7B, and Gemma-2 9B.
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import ModelLoader
from src.benchmark import LLMBenchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Benchmark LLM inference time for Softmax vs other operations'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=['llama2-7b', 'llama3.1-8b', 'mistral-7b', 'gemma2-9b'],
        choices=['llama2-7b', 'llama3.1-8b', 'mistral-7b', 'gemma2-9b', 'all'],
        help='Models to benchmark (default: all)'
    )

    parser.add_argument(
        '--context-length',
        type=int,
        default=8192,
        help='Context length in tokens (default: 8192)'
    )

    parser.add_argument(
        '--num-tokens',
        type=int,
        default=1024,
        help='Number of tokens to generate (default: 1024)'
    )

    parser.add_argument(
        '--num-runs',
        type=int,
        default=1,
        help='Number of benchmark runs to average (default: 1)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on (default: cuda)'
    )

    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float16', 'float32', 'bfloat16'],
        help='Model dtype (default: float16)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )

    parser.add_argument(
        '--flash-attention',
        action='store_true',
        help='Use Flash Attention 2 if available'
    )

    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype."""
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    return dtype_map[dtype_str]


def save_results(results: dict, output_dir: str, model_key: str):
    """Save benchmark results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_key}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results, indent=2, fp=f)

    logger.info(f"Results saved to {filepath}")
    return filepath


def print_summary_table(all_results: dict):
    """Print a summary table of all results."""
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY")
    print("="*100)
    print(f"{'Model':<20} {'Total (s)':<12} {'Softmax (s)':<12} {'Other (s)':<12} {'Softmax %':<12} {'Tokens/s':<12}")
    print("-"*100)

    for model_key, results in all_results.items():
        model_name = ModelLoader.get_model_info(model_key)['name']
        print(
            f"{model_name:<20} "
            f"{results['total_time']:<12.4f} "
            f"{results['softmax_time']:<12.4f} "
            f"{results['other_time']:<12.4f} "
            f"{results['softmax_percentage']:<12.2f} "
            f"{results['tokens_per_second']:<12.2f}"
        )

    print("="*100)


def main():
    """Main function to run benchmarks."""
    args = parse_args()

    # Handle 'all' models option
    if 'all' in args.models:
        models_to_benchmark = ['llama2-7b', 'llama3.1-8b', 'mistral-7b', 'gemma2-9b']
    else:
        models_to_benchmark = args.models

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
        args.dtype = 'float32'

    dtype = get_dtype(args.dtype)

    logger.info("="*100)
    logger.info("LLM Softmax Benchmark")
    logger.info("="*100)
    logger.info(f"Models: {', '.join(models_to_benchmark)}")
    logger.info(f"Context length: {args.context_length}")
    logger.info(f"Tokens to generate: {args.num_tokens}")
    logger.info(f"Number of runs: {args.num_runs}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Dtype: {args.dtype}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*100)

    all_results = {}

    for model_key in models_to_benchmark:
        try:
            logger.info(f"\n{'='*100}")
            logger.info(f"Benchmarking {ModelLoader.get_model_info(model_key)['name']}")
            logger.info(f"{'='*100}")

            # Load model
            model, tokenizer = ModelLoader.load_model(
                model_key,
                device=args.device,
                torch_dtype=dtype,
                use_flash_attention=args.flash_attention
            )

            # Create benchmark
            benchmark = LLMBenchmark(model, tokenizer, device=args.device)

            # Run benchmark
            results = benchmark.run_benchmark(
                context_length=args.context_length,
                num_tokens=args.num_tokens,
                num_runs=args.num_runs
            )

            # Add metadata
            results['model_key'] = model_key
            results['model_name'] = ModelLoader.get_model_info(model_key)['name']
            results['device'] = args.device
            results['dtype'] = args.dtype
            results['timestamp'] = datetime.now().isoformat()

            # Save results
            save_results(results, args.output_dir, model_key)

            all_results[model_key] = results

            # Cleanup
            ModelLoader.cleanup_model(model)
            del tokenizer
            del benchmark

        except Exception as e:
            logger.error(f"Error benchmarking {model_key}: {e}", exc_info=True)
            continue

    # Print summary
    if all_results:
        print_summary_table(all_results)

        # Save combined results
        combined_results = {
            'benchmark_config': {
                'context_length': args.context_length,
                'num_tokens': args.num_tokens,
                'num_runs': args.num_runs,
                'device': args.device,
                'dtype': args.dtype
            },
            'results': all_results
        }

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        combined_filepath = os.path.join(args.output_dir, f'combined_results_{timestamp}.json')
        with open(combined_filepath, 'w') as f:
            json.dump(combined_results, indent=2, fp=f)

        logger.info(f"\nCombined results saved to {combined_filepath}")
    else:
        logger.error("No successful benchmark runs")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
