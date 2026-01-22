# LLM Softmax Time Benchmark

A comprehensive benchmarking suite for measuring the time consumed by Softmax operations versus all other operations during LLM inference. This tool helps analyze the computational breakdown of different language models during token generation.

## Features

- **Precise Time Profiling**: Separates Softmax operation time from other operations using PyTorch hooks
- **Multiple Model Support**: Benchmarks LLaMA-2 7B, LLaMA-3.1 8B, Mistral 7B, and Gemma-2 9B
- **Configurable Parameters**: Customize context length, generation length, and number of runs
- **Detailed Results**: JSON output with comprehensive timing statistics
- **Analysis Tools**: Built-in tools for analyzing and comparing results

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, but CPU is supported)
- At least 16GB GPU memory for 7B models, 24GB+ recommended for larger models
- Hugging Face account with access to gated models (LLaMA, Gemma)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SoftmaxTimeBenchmark.git
cd SoftmaxTimeBenchmark
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Login to Hugging Face (required for gated models):
```bash
huggingface-cli login
```

You'll need access to:
- `meta-llama/Llama-2-7b-hf`
- `meta-llama/Llama-3.1-8B`
- `google/gemma-2-9b`

Request access at: https://huggingface.co/meta-llama and https://huggingface.co/google

## Quick Start

### Run Full Benchmark (All Models)

```bash
python run_benchmark.py
```

This will benchmark all four models with default parameters:
- Context length: 8192 tokens
- Tokens to generate: 1024
- Number of runs: 1

### Run Quick Test

For a quick verification that everything works:

```bash
bash quick_test.sh
```

This runs a reduced benchmark (512 context, 128 tokens) on LLaMA-2 7B.

## Usage

### Basic Usage

```bash
python run_benchmark.py --models llama2-7b mistral-7b
```

### Advanced Options

```bash
python run_benchmark.py \
    --models llama2-7b llama3.1-8b mistral-7b gemma2-9b \
    --context-length 8192 \
    --num-tokens 1024 \
    --num-runs 3 \
    --device cuda \
    --dtype float16 \
    --output-dir results
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--models` | Models to benchmark (llama2-7b, llama3.1-8b, mistral-7b, gemma2-9b, all) | all |
| `--context-length` | Context length in tokens | 8192 |
| `--num-tokens` | Number of tokens to generate | 1024 |
| `--num-runs` | Number of runs to average | 1 |
| `--device` | Device to use (cuda, cpu) | cuda |
| `--dtype` | Model dtype (float16, float32, bfloat16) | float16 |
| `--output-dir` | Output directory for results | results |
| `--flash-attention` | Use Flash Attention 2 if available | False |

## Example Commands

### Benchmark a Single Model

```bash
python run_benchmark.py --models llama2-7b
```

### Benchmark with CPU

```bash
python run_benchmark.py --device cpu --dtype float32
```

### Multiple Runs for Averaging

```bash
python run_benchmark.py --num-runs 5
```

### Custom Context and Generation Length

```bash
python run_benchmark.py \
    --context-length 4096 \
    --num-tokens 512
```

### Benchmark Specific Models Only

```bash
python run_benchmark.py --models llama2-7b mistral-7b
```

## Understanding Results

### Output Files

Results are saved in the `results/` directory:
- `{model_key}_{timestamp}.json`: Individual model results
- `combined_results_{timestamp}.json`: All models combined

### Result Format

```json
{
  "total_time": 45.23,           // Total inference time (seconds)
  "softmax_time": 2.15,          // Time in Softmax operations (seconds)
  "other_time": 43.08,           // Time in other operations (seconds)
  "softmax_percentage": 4.75,    // Percentage of time in Softmax
  "tokens_per_second": 22.63,    // Generation speed
  "softmax_count": 12288,        // Number of Softmax calls
  "model_name": "LLaMA-2 7B",
  "context_length": 8192,
  "num_tokens": 1024
}
```

### Analyzing Results

Use the analysis script to view results:

```bash
# Analyze a single result file
python analyze_results.py results/llama2-7b_20240115_143022.json

# Analyze combined results
python analyze_results.py results/combined_results_20240115_143022.json

# Compare multiple results
python analyze_results.py --compare results/*.json
```

## Architecture

### Project Structure

```
SoftmaxTimeBenchmark/
├── src/
│   ├── __init__.py
│   ├── profiler.py       # Time profiling with PyTorch hooks
│   ├── model_loader.py   # Model loading utilities
│   └── benchmark.py      # Benchmarking logic
├── run_benchmark.py      # Main benchmark script
├── analyze_results.py    # Results analysis tool
├── quick_test.sh         # Quick test script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### How It Works

1. **Profiler**: Monkey-patches `torch.nn.functional.softmax` to intercept and time all Softmax calls
2. **Model Loader**: Loads models from Hugging Face with optimized settings
3. **Benchmark**: Generates tokens while profiling, measuring total time and Softmax time separately
4. **Results**: Computes statistics including percentage of time spent in Softmax operations

## Benchmarking Details

### Test Configuration

The default benchmark:
1. Loads the model in float16 precision
2. Prepares a context of 8192 tokens (repeated text)
3. Generates 1024 new tokens
4. Measures time for each Softmax operation
5. Computes total time and percentage breakdown

### What is Measured

- **Softmax Time**: Time spent in `torch.nn.functional.softmax()` operations, including:
  - Attention softmax in self-attention layers
  - Final softmax over vocabulary for token sampling

- **Other Operations**: All other operations, including:
  - Matrix multiplications
  - Layer normalization
  - Activation functions (GELU, SiLU)
  - Embedding lookups
  - All other transformer operations

### Performance Tips

1. **Use GPU**: CUDA is much faster than CPU
2. **Use float16/bfloat16**: Reduces memory and increases speed
3. **Enable Flash Attention**: Use `--flash-attention` if available
4. **Multiple Runs**: Use `--num-runs 3` or more for stable averages
5. **Monitor GPU**: Watch `nvidia-smi` to ensure you're not running out of memory

## Troubleshooting

### Out of Memory (OOM)

If you get OOM errors:
- Reduce `--context-length` (try 4096 or 2048)
- Reduce `--num-tokens` (try 512)
- Use `--dtype float16` (smaller than float32)
- Try one model at a time

### Model Access Issues

If you can't load models:
- Make sure you've logged in: `huggingface-cli login`
- Request access to gated models on Hugging Face
- Check your internet connection

### Slow Performance

- Use GPU instead of CPU: `--device cuda`
- Use float16: `--dtype float16`
- Enable Flash Attention: `--flash-attention`

## Expected Results

Typical Softmax percentage ranges (may vary):
- **LLaMA-2 7B**: 3-6% of total time
- **LLaMA-3.1 8B**: 3-6% of total time
- **Mistral 7B**: 3-6% of total time
- **Gemma-2 9B**: 3-6% of total time

Most time is spent in:
1. Matrix multiplications (40-60%)
2. Memory operations (20-30%)
3. Activation functions (10-15%)
4. Softmax (3-6%)
5. Other operations (5-10%)

## Citation

If you use this benchmarking suite in your research, please cite:

```bibtex
@software{softmax_time_benchmark,
  title = {LLM Softmax Time Benchmark},
  year = {2024},
  url = {https://github.com/yourusername/SoftmaxTimeBenchmark}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or issues, please open a GitHub issue.
