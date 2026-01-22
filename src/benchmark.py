"""
Main benchmarking module for measuring LLM inference time.
Generates tokens and profiles time spent in Softmax vs other operations.
"""

import torch
import time
import logging
from typing import Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .profiler import AttentionProfiler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMBenchmark:
    """
    Benchmark class for measuring LLM inference performance.
    Specifically measures time spent in Softmax operations vs other operations.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = 'cuda',
        attn_implementation: str = 'eager'
    ):
        """
        Initialize the benchmark.

        Args:
            model: The language model to benchmark
            tokenizer: The tokenizer for the model
            device: Device to run on ('cuda' or 'cpu')
            attn_implementation: Attention implementation used by model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.attn_implementation = attn_implementation
        self.profiler = AttentionProfiler(attention_type=attn_implementation)

    def prepare_context(self, context_length: int = 8192) -> torch.Tensor:
        """
        Prepare a context of specified length.

        Args:
            context_length: Target context length in tokens

        Returns:
            Tensor of input IDs
        """
        # Create a meaningful prompt that will be repeated to reach context length
        prompt = (
            "The following is a detailed technical discussion about artificial intelligence "
            "and machine learning. We will explore various aspects of neural networks, "
            "transformer architectures, and their applications in natural language processing. "
        )

        # Tokenize the base prompt
        base_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)

        # Repeat to reach desired context length
        num_repeats = (context_length // len(base_tokens)) + 1
        tokens = base_tokens * num_repeats
        tokens = tokens[:context_length]

        logger.info(f"Prepared context with {len(tokens)} tokens")

        # Convert to tensor
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        return input_ids

    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        num_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> Dict[str, float]:
        """
        Generate tokens and measure timing.

        Args:
            input_ids: Input context tensor
            num_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Dictionary with timing results
        """
        logger.info(f"Generating {num_tokens} tokens...")

        # Warm up
        logger.info("Warming up model...")
        with torch.no_grad():
            _ = self.model(input_ids[:, :10])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Start profiling
        generated_tokens = 0
        current_input_ids = input_ids

        with self.profiler.profile():
            with torch.no_grad():
                for i in range(num_tokens):
                    # Forward pass
                    outputs = self.model(current_input_ids)
                    logits = outputs.logits[:, -1, :]

                    # Apply temperature
                    logits = logits / temperature

                    # Apply softmax to get probabilities (this is what we're measuring!)
                    probs = torch.nn.functional.softmax(logits, dim=-1)

                    # Sample next token
                    if top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                        probs = torch.nn.functional.softmax(logits, dim=-1)

                    next_token = torch.multinomial(probs, num_samples=1)

                    # Append to sequence
                    current_input_ids = torch.cat([current_input_ids, next_token], dim=1)

                    generated_tokens += 1

                    if (i + 1) % 100 == 0:
                        logger.info(f"Generated {i + 1}/{num_tokens} tokens")

        logger.info(f"Successfully generated {generated_tokens} tokens")

        # Get profiling results
        results = self.profiler.get_summary()
        return results

    def run_benchmark(
        self,
        context_length: int = 8192,
        num_tokens: int = 1024,
        num_runs: int = 1
    ) -> Dict[str, any]:
        """
        Run complete benchmark.

        Args:
            context_length: Context length in tokens
            num_tokens: Number of tokens to generate
            num_runs: Number of benchmark runs to average

        Returns:
            Dictionary with benchmark results
        """
        logger.info("="*80)
        logger.info(f"Starting benchmark: {context_length} context, {num_tokens} tokens, {num_runs} runs")
        logger.info("="*80)

        all_results = []

        for run in range(num_runs):
            logger.info(f"\nRun {run + 1}/{num_runs}")
            logger.info("-"*80)

            # Prepare context
            input_ids = self.prepare_context(context_length)

            # Run generation with profiling
            results = self.generate_tokens(input_ids, num_tokens)

            all_results.append(results)

            # Clean up
            del input_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Average results
        avg_results = {
            'softmax_time': sum(r['softmax_time'] for r in all_results) / num_runs,
            'other_time': sum(r['other_time'] for r in all_results) / num_runs,
            'total_time': sum(r['total_time'] for r in all_results) / num_runs,
            'softmax_percentage': sum(r['softmax_percentage'] for r in all_results) / num_runs,
            'softmax_count': sum(r['softmax_count'] for r in all_results) / num_runs,
            'context_length': context_length,
            'num_tokens': num_tokens,
            'num_runs': num_runs
        }

        # Calculate tokens per second
        avg_results['tokens_per_second'] = num_tokens / avg_results['total_time']

        logger.info("\n" + "="*80)
        logger.info("Benchmark Results:")
        logger.info("="*80)
        logger.info(f"Total time: {avg_results['total_time']:.4f} seconds")
        logger.info(f"Softmax time: {avg_results['softmax_time']:.4f} seconds ({avg_results['softmax_percentage']:.2f}%)")
        logger.info(f"Other operations time: {avg_results['other_time']:.4f} seconds ({100-avg_results['softmax_percentage']:.2f}%)")
        logger.info(f"Tokens per second: {avg_results['tokens_per_second']:.2f}")
        logger.info(f"Softmax calls: {avg_results['softmax_count']:.0f}")
        logger.info("="*80)

        return avg_results
