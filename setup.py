"""Setup script for SoftmaxTimeBenchmark package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="softmax-time-benchmark",
    version="1.0.0",
    author="SoftmaxTimeBenchmark Contributors",
    description="Benchmark suite for measuring Softmax vs other operations in LLM inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SoftmaxTimeBenchmark",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "softmax-benchmark=run_benchmark:main",
        ],
    },
)
