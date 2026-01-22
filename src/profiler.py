"""
Profiler module for measuring time spent in Softmax and other operations.
Handles different attention implementations (eager, SDPA, Flash Attention).
"""

import time
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class AttentionProfiler:
    """
    Advanced profiler that handles different attention implementations in transformers.

    Transformers can use different attention mechanisms:
    - eager: Uses explicit F.softmax calls (easiest to profile)
    - sdpa: Uses fused torch.nn.functional.scaled_dot_product_attention
    - flash_attention_2: Uses Flash Attention 2 (fully fused kernel)

    This profiler patches the relevant functions to measure softmax time.
    """

    def __init__(self, attention_type: str = 'auto'):
        """
        Initialize profiler.

        Args:
            attention_type: 'auto', 'eager', 'sdpa', or 'flash_attention_2'
        """
        self.attention_type = attention_type
        self.softmax_time = 0.0
        self.softmax_count = 0
        self.sdpa_time = 0.0
        self.sdpa_count = 0
        self.total_time = 0.0
        self.enabled = False

        # Store original functions
        self._original_softmax = None
        self._original_sdpa = None
        self._start_time = None

        # Detailed operation tracking
        self.operation_times = defaultdict(float)
        self.operation_counts = defaultdict(int)

    def reset(self):
        """Reset all timing statistics."""
        self.softmax_time = 0.0
        self.softmax_count = 0
        self.sdpa_time = 0.0
        self.sdpa_count = 0
        self.total_time = 0.0
        self._start_time = None
        self.operation_times.clear()
        self.operation_counts.clear()

    def _patched_softmax(self, *args, **kwargs):
        """Patched version of F.softmax that measures execution time."""
        if not self.enabled:
            return self._original_softmax(*args, **kwargs)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        result = self._original_softmax(*args, **kwargs)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start

        self.softmax_time += elapsed
        self.softmax_count += 1
        self.operation_times['softmax'] += elapsed
        self.operation_counts['softmax'] += 1

        return result

    def _patched_sdpa(self, *args, **kwargs):
        """
        Patched version of scaled_dot_product_attention.

        SDPA fuses the entire attention computation including softmax.
        We measure the total SDPA time and estimate softmax contribution.
        """
        if not self.enabled:
            return self._original_sdpa(*args, **kwargs)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        result = self._original_sdpa(*args, **kwargs)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start

        self.sdpa_time += elapsed
        self.sdpa_count += 1
        self.operation_times['sdpa'] += elapsed
        self.operation_counts['sdpa'] += 1

        # Estimate softmax time as a portion of SDPA time
        # Based on typical attention breakdown: QK^T (40%), softmax (10%), softmax*V (50%)
        estimated_softmax = elapsed * 0.1
        self.softmax_time += estimated_softmax
        self.softmax_count += 1

        return result

    def patch_functions(self):
        """Monkey-patch attention-related functions."""
        # Always patch F.softmax
        if self._original_softmax is None:
            self._original_softmax = F.softmax
            F.softmax = self._patched_softmax
            logger.info("Patched F.softmax")

        # Patch scaled_dot_product_attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            if self._original_sdpa is None:
                self._original_sdpa = F.scaled_dot_product_attention
                F.scaled_dot_product_attention = self._patched_sdpa
                logger.info("Patched F.scaled_dot_product_attention")
        else:
            logger.warning("F.scaled_dot_product_attention not available (PyTorch < 2.0)")

    def unpatch_functions(self):
        """Restore original functions."""
        if self._original_softmax is not None:
            F.softmax = self._original_softmax
            self._original_softmax = None
            logger.debug("Unpatched F.softmax")

        if self._original_sdpa is not None:
            F.scaled_dot_product_attention = self._original_sdpa
            self._original_sdpa = None
            logger.debug("Unpatched F.scaled_dot_product_attention")

    @contextmanager
    def profile(self):
        """Context manager to enable profiling."""
        self.reset()
        self.patch_functions()
        self.enabled = True
        self._start_time = time.perf_counter()

        try:
            yield self
        finally:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            self.total_time = time.perf_counter() - self._start_time
            self.enabled = False

            # Log detection results
            if self.softmax_count == 0 and self.sdpa_count == 0:
                logger.warning(
                    "No softmax or SDPA operations detected! "
                    "The model may be using Flash Attention or another fused implementation. "
                    "Consider using --attn-implementation eager for accurate softmax profiling."
                )
            elif self.sdpa_count > 0:
                logger.info(
                    f"Detected SDPA usage ({self.sdpa_count} calls). "
                    f"Softmax time is estimated at ~10% of SDPA time. "
                    f"Use --attn-implementation eager for exact measurements."
                )

    def get_summary(self) -> Dict[str, any]:
        """
        Get a summary of profiling results.

        Returns:
            Dictionary with timing statistics
        """
        other_time = self.total_time - self.softmax_time
        softmax_percentage = (self.softmax_time / self.total_time * 100) if self.total_time > 0 else 0

        # Determine which attention implementation was used
        if self.sdpa_count > 0 and self.softmax_count <= self.sdpa_count:
            attention_impl = 'sdpa (softmax time estimated)'
        elif self.softmax_count > 0:
            attention_impl = 'eager (softmax time exact)'
        else:
            attention_impl = 'unknown (possibly flash_attention_2)'

        return {
            'softmax_time': self.softmax_time,
            'other_time': other_time,
            'total_time': self.total_time,
            'softmax_percentage': softmax_percentage,
            'softmax_count': self.softmax_count,
            'sdpa_count': self.sdpa_count,
            'sdpa_time': self.sdpa_time,
            'attention_implementation': attention_impl,
            'operation_times': dict(self.operation_times),
            'operation_counts': dict(self.operation_counts)
        }

    def __del__(self):
        """Ensure functions are unpatched when profiler is destroyed."""
        self.unpatch_functions()


# Backwards compatibility alias
FunctionalSoftmaxProfiler = AttentionProfiler
