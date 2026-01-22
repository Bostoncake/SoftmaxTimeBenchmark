#!/usr/bin/env python3
"""
Script to analyze and visualize benchmark results.
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Dict


def load_results(filepath: str) -> dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_single_result(filepath: str):
    """Analyze a single result file."""
    results = load_results(filepath)

    print("\n" + "="*80)
    print(f"Analysis of {Path(filepath).name}")
    print("="*80)

    if 'model_name' in results:
        # Single model result
        print(f"\nModel: {results['model_name']}")
        print(f"Context Length: {results['context_length']}")
        print(f"Tokens Generated: {results['num_tokens']}")
        print(f"Number of Runs: {results['num_runs']}")
        print(f"\nTiming Results:")
        print(f"  Total Time: {results['total_time']:.4f} seconds")
        print(f"  Softmax Time: {results['softmax_time']:.4f} seconds ({results['softmax_percentage']:.2f}%)")
        print(f"  Other Operations: {results['other_time']:.4f} seconds ({100-results['softmax_percentage']:.2f}%)")
        print(f"  Tokens/Second: {results['tokens_per_second']:.2f}")
        print(f"  Softmax Calls: {results['softmax_count']:.0f}")

    elif 'results' in results:
        # Combined results
        print("\nCombined Results Analysis")
        print(f"\nConfiguration:")
        config = results['benchmark_config']
        print(f"  Context Length: {config['context_length']}")
        print(f"  Tokens Generated: {config['num_tokens']}")
        print(f"  Number of Runs: {config['num_runs']}")
        print(f"  Device: {config['device']}")
        print(f"  Dtype: {config['dtype']}")

        print("\n" + "-"*80)
        print(f"{'Model':<20} {'Total (s)':<12} {'Softmax (s)':<12} {'Other (s)':<12} {'Softmax %':<12} {'Tokens/s':<12}")
        print("-"*80)

        for model_key, model_results in results['results'].items():
            print(
                f"{model_results['model_name']:<20} "
                f"{model_results['total_time']:<12.4f} "
                f"{model_results['softmax_time']:<12.4f} "
                f"{model_results['other_time']:<12.4f} "
                f"{model_results['softmax_percentage']:<12.2f} "
                f"{model_results['tokens_per_second']:<12.2f}"
            )

        # Comparison insights
        print("\n" + "-"*80)
        print("Insights:")

        model_results_list = list(results['results'].values())

        # Find model with highest/lowest softmax percentage
        max_softmax_model = max(model_results_list, key=lambda x: x['softmax_percentage'])
        min_softmax_model = min(model_results_list, key=lambda x: x['softmax_percentage'])

        print(f"\nHighest Softmax %: {max_softmax_model['model_name']} ({max_softmax_model['softmax_percentage']:.2f}%)")
        print(f"Lowest Softmax %: {min_softmax_model['model_name']} ({min_softmax_model['softmax_percentage']:.2f}%)")

        # Find fastest model
        fastest_model = max(model_results_list, key=lambda x: x['tokens_per_second'])
        print(f"\nFastest Model: {fastest_model['model_name']} ({fastest_model['tokens_per_second']:.2f} tokens/s)")

        # Average softmax percentage
        avg_softmax_pct = sum(r['softmax_percentage'] for r in model_results_list) / len(model_results_list)
        print(f"\nAverage Softmax %: {avg_softmax_pct:.2f}%")

    print("="*80 + "\n")


def compare_results(filepaths: List[str]):
    """Compare multiple result files."""
    all_results = []

    for filepath in filepaths:
        results = load_results(filepath)
        if 'model_name' in results:
            all_results.append(results)

    if not all_results:
        print("No valid result files to compare")
        return

    print("\n" + "="*80)
    print("Results Comparison")
    print("="*80)
    print(f"{'Model':<20} {'Total (s)':<12} {'Softmax (s)':<12} {'Softmax %':<12} {'Tokens/s':<12}")
    print("-"*80)

    for results in all_results:
        print(
            f"{results['model_name']:<20} "
            f"{results['total_time']:<12.4f} "
            f"{results['softmax_time']:<12.4f} "
            f"{results['softmax_percentage']:<12.2f} "
            f"{results['tokens_per_second']:<12.2f}"
        )

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument(
        'files',
        nargs='+',
        help='Result files to analyze'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple files'
    )

    args = parser.parse_args()

    if args.compare:
        compare_results(args.files)
    else:
        for filepath in args.files:
            analyze_single_result(filepath)


if __name__ == '__main__':
    main()
