# Contributing to SoftmaxTimeBenchmark

Thank you for your interest in contributing to SoftmaxTimeBenchmark!

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in the GitHub Issues
2. If not, create a new issue with:
   - Clear description of the problem/feature
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (Python version, GPU model, etc.)

### Submitting Changes

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Test your changes thoroughly
5. Commit with clear messages: `git commit -m "Add feature: description"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Submit a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and concise
- Add comments for complex logic

### Testing

Before submitting:

1. Test with at least one model: `bash quick_test.sh`
2. Verify no syntax errors: `python -m py_compile src/*.py`
3. Check that results are reasonable

### Adding New Models

To add support for a new model:

1. Add model configuration to `src/model_loader.py` in `MODEL_CONFIGS`
2. Test that the model loads correctly
3. Run a benchmark to verify it works
4. Update README.md with the new model

### Documentation

- Update README.md for new features
- Add docstrings to new functions
- Include usage examples
- Update CHANGELOG.md

## Questions?

Open an issue for questions or discussions.

Thank you for contributing!
