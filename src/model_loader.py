"""
Model loader module for loading different LLM models.
Supports LLaMA-2, LLaMA-3.1, Mistral, and Gemma-2 models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model configurations
MODEL_CONFIGS = {
    'llama2-7b': {
        'model_id': 'meta-llama/Llama-2-7b-hf',
        'name': 'LLaMA-2 7B'
    },
    'llama3.1-8b': {
        'model_id': 'meta-llama/Llama-3.1-8B',
        'name': 'LLaMA-3.1 8B'
    },
    'mistral-7b': {
        'model_id': 'mistralai/Mistral-7B-v0.1',
        'name': 'Mistral 7B'
    },
    'gemma2-9b': {
        'model_id': 'google/gemma-2-9b',
        'name': 'Gemma-2 9B'
    }
}


class ModelLoader:
    """Utility class for loading and managing LLM models."""

    @staticmethod
    def load_model(
        model_key: str,
        device: str = 'cuda',
        torch_dtype: torch.dtype = torch.float16,
        use_flash_attention: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a model and tokenizer.

        Args:
            model_key: Key from MODEL_CONFIGS (e.g., 'llama2-7b')
            device: Device to load model on ('cuda' or 'cpu')
            torch_dtype: Data type for model weights
            use_flash_attention: Whether to use Flash Attention 2 if available

        Returns:
            Tuple of (model, tokenizer)
        """
        if model_key not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model key: {model_key}. "
                f"Available keys: {list(MODEL_CONFIGS.keys())}"
            )

        config = MODEL_CONFIGS[model_key]
        model_id = config['model_id']
        name = config['name']

        logger.info(f"Loading {name} from {model_id}...")

        # Check if CUDA is available
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'
            torch_dtype = torch.float32

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        logger.info(f"Loading model on {device} with dtype {torch_dtype}...")
        model_kwargs = {
            'pretrained_model_name_or_path': model_id,
            'torch_dtype': torch_dtype,
            'device_map': 'auto' if device == 'cuda' else None,
            'trust_remote_code': True,
            'low_cpu_mem_usage': True
        }

        # Add Flash Attention if requested
        if use_flash_attention:
            try:
                model_kwargs['attn_implementation'] = 'flash_attention_2'
                logger.info("Using Flash Attention 2")
            except Exception as e:
                logger.warning(f"Flash Attention 2 not available: {e}")

        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        if device == 'cpu':
            model = model.to(device)

        model.eval()

        logger.info(f"Successfully loaded {name}")
        return model, tokenizer

    @staticmethod
    def get_model_info(model_key: str) -> dict:
        """Get information about a model."""
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}")
        return MODEL_CONFIGS[model_key]

    @staticmethod
    def list_available_models() -> list:
        """List all available model keys."""
        return list(MODEL_CONFIGS.keys())

    @staticmethod
    def cleanup_model(model: AutoModelForCausalLM):
        """Clean up model from memory."""
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model cleaned up from memory")
