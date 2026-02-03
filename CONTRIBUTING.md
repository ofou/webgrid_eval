# Contributing to Webgrid Eval

Thank you for your interest in contributing to Webgrid Eval! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report, please check the existing issues to see if the problem has already been reported. When you create a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and what behavior you expected**
- **Include code samples or screenshots if relevant**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the enhancement**
- **Explain why this enhancement would be useful**

### Pull Requests

1. Fork the repository
2. Create a new branch from `main` for your feature or bug fix
3. Make your changes
4. Ensure your code follows the style guidelines
5. Add or update tests as necessary
6. Update documentation as needed
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.10 or higher
- uv (recommended) or pip

### Installation

```bash
# Clone your fork
git clone https://github.com/omarolivares/webgrid_eval.git
cd webgrid_eval

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=webgrid_eval
```

### Code Style

This project uses:
- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **MyPy** for type checking

```bash
# Format code
black webgrid_eval tests

# Run linter
ruff check webgrid_eval tests

# Run type checker
mypy webgrid_eval
```

## Project Structure

```
webgrid_eval/
├── webgrid_eval/        # Main package
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── openrouter.py    # LLM client and agent loop
│   ├── game_state.py    # Game state management
│   ├── tools.py         # Tool definitions
│   ├── screenshot.py    # Screenshot generation
│   ├── run_eval.py      # CLI evaluation runner
│   ├── make_gif.py      # GIF generation
│   └── assets/          # Static assets
├── tests/               # Test suite
├── examples/            # Usage examples
├── configs/             # Configuration files
├── docs/                # Documentation
└── results/             # Evaluation results (gitignored)
```

## Adding New Features

### Adding a New Metric

If you want to add a new evaluation metric:

1. Add the metric calculation to the appropriate module
2. Update the result models in `main.py`
3. Add tests for the metric
4. Update documentation

### Adding Support for New Model Providers

To add support for a new LLM provider:

1. Update `openrouter.py` to handle the provider's API
2. Add configuration examples to `configs/`
3. Document the provider in the README

## Testing Guidelines

- Write tests for all new functionality
- Aim for high test coverage
- Use pytest fixtures for common setup
- Mock external API calls in tests

## Documentation

- Update the README.md if you change functionality
- Add docstrings to all public functions and classes
- Follow Google-style docstrings
- Keep configuration examples up to date

## Commit Messages

Use clear and meaningful commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests when relevant

Example:
```
Add support for custom grid sizes

- Implement dynamic grid sizing in GameState
- Update screenshot rendering for variable sizes
- Add configuration option for grid size

Fixes #123
```

## Questions?

Feel free to open an issue for questions or join discussions.

Thank you for contributing!
