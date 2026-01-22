"""
Profiler module for measuring time spent in Softmax and other operations.
Uses PyTorch hooks to intercept and time operations during forward pass.
"""

import time
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Tuple
from contextlib import contextmanager


class OperationProfiler:
    """
    Profiler that hooks into PyTorch operations to measure execution time.
    Specifically tracks Softmax operations separately from other operations.
    """

    def __init__(self):
        self.softmax_time = 0.0
        self.total_time = 0.0
        self.operation_counts = defaultdict(int)
        self.operation_times = defaultdict(float)
        self.hooks = []
        self.enabled = False

    def reset(self):
        """Reset all timing statistics."""
        self.softmax_time = 0.0
        self.total_time = 0.0
        self.operation_counts.clear()
        self.operation_times.clear()

    def _create_forward_hook(self, module_name: str):
        """Create a forward hook for a module to measure execution time."""
        def hook(module, input, output):
            if not self.enabled:
                return

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()

            # Let the output be computed (already done, but we measure the time)
            # We actually need to measure during execution, so we'll use a different approach

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - start_time

            self.operation_counts[module_name] += 1
            self.operation_times[module_name] += elapsed

        return hook

    def _create_pre_forward_hook(self, module_name: str):
        """Create a pre-forward hook to mark the start of execution."""
        def hook(module, input):
            if not self.enabled:
                return
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            module._profiler_start_time = time.perf_counter()
        return hook

    def _create_post_forward_hook(self, module_name: str, is_softmax: bool = False):
        """Create a post-forward hook to measure execution time."""
        def hook(module, input, output):
            if not self.enabled:
                return

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - module._profiler_start_time

            self.operation_counts[module_name] += 1
            self.operation_times[module_name] += elapsed

            if is_softmax:
                self.softmax_time += elapsed
            self.total_time += elapsed

        return hook

    def register_hooks(self, model: torch.nn.Module):
        """
        Register hooks on all modules in the model.
        Identifies Softmax operations and tracks them separately.
        """
        self.remove_hooks()

        for name, module in model.named_modules():
            # Check if this is a Softmax operation
            is_softmax = isinstance(module, torch.nn.Softmax)

            # Register pre and post hooks
            pre_hook = module.register_forward_pre_hook(
                self._create_pre_forward_hook(name)
            )
            post_hook = module.register_forward_hook(
                self._create_post_forward_hook(name, is_softmax)
            )

            self.hooks.append(pre_hook)
            self.hooks.append(post_hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    @contextmanager
    def profile(self):
        """Context manager to enable profiling."""
        self.reset()
        self.enabled = True
        try:
            yield self
        finally:
            self.enabled = False

    def get_summary(self) -> Dict[str, float]:
        """
        Get a summary of profiling results.

        Returns:
            Dictionary with timing statistics:
            - softmax_time: Total time spent in Softmax operations
            - other_time: Total time spent in non-Softmax operations
            - total_time: Total time for all operations
            - softmax_percentage: Percentage of time spent in Softmax
        """
        other_time = self.total_time - self.softmax_time
        softmax_percentage = (self.softmax_time / self.total_time * 100) if self.total_time > 0 else 0

        return {
            'softmax_time': self.softmax_time,
            'other_time': other_time,
            'total_time': self.total_time,
            'softmax_percentage': softmax_percentage,
            'operation_counts': dict(self.operation_counts),
            'operation_times': dict(self.operation_times)
        }


class FunctionalSoftmaxProfiler:
    """
    Alternative profiler that monkey-patches F.softmax to measure functional softmax calls.
    This catches softmax operations that aren't nn.Softmax modules.
    """

    def __init__(self):
        self.softmax_time = 0.0
        self.softmax_count = 0
        self.total_time = 0.0
        self.enabled = False
        self._original_softmax = None
        self._start_time = None

    def reset(self):
        """Reset all timing statistics."""
        self.softmax_time = 0.0
        self.softmax_count = 0
        self.total_time = 0.0
        self._start_time = None

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

        return result

    def patch_softmax(self):
        """Monkey-patch F.softmax to intercept calls."""
        if self._original_softmax is None:
            self._original_softmax = F.softmax
            F.softmax = self._patched_softmax

    def unpatch_softmax(self):
        """Restore original F.softmax."""
        if self._original_softmax is not None:
            F.softmax = self._original_softmax
            self._original_softmax = None

    @contextmanager
    def profile(self):
        """Context manager to enable profiling."""
        self.reset()
        self.patch_softmax()
        self.enabled = True
        self._start_time = time.perf_counter()

        try:
            yield self
        finally:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            self.total_time = time.perf_counter() - self._start_time
            self.enabled = False

    def get_summary(self) -> Dict[str, float]:
        """Get a summary of profiling results."""
        other_time = self.total_time - self.softmax_time
        softmax_percentage = (self.softmax_time / self.total_time * 100) if self.total_time > 0 else 0

        return {
            'softmax_time': self.softmax_time,
            'other_time': other_time,
            'total_time': self.total_time,
            'softmax_percentage': softmax_percentage,
            'softmax_count': self.softmax_count
        }

    def __del__(self):
        """Ensure softmax is unpatched when profiler is destroyed."""
        self.unpatch_softmax()
