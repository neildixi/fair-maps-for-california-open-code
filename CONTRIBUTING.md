# Contributing to FairMaps

Thank you for your interest in contributing to FairMaps. This document explains how you can help.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an [Issue](https://github.com/YOUR_ORG/fair-maps-for-california-open-code/issues) with:

- A clear description of the problem
- Steps to reproduce
- Your environment (Python version, OS)
- Any error messages or screenshots

### Suggesting Features

Feature requests are welcome. Open an Issue and describe:

- The use case or problem you want to solve
- Your proposed solution or idea
- Any alternatives you considered

### Pull Requests

1. **Fork** the repository and clone your fork
2. **Create a branch**: `git checkout -b feature/your-feature-name` or `fix/your-bug-fix`
3. **Install in development mode**: `pip install -e ".[dev]"`
4. **Make your changes** and ensure tests pass: `pytest tests/ -v`
5. **Commit** with clear messages
6. **Push** to your fork and open a Pull Request

We ask that:

- Code follows the existing style (PEP 8)
- New features include tests where possible
- The PR description explains the change and why it's needed

### Getting Help

- **Questions**: Open a [Discussion](https://github.com/YOUR_ORG/fair-maps-for-california-open-code/discussions) or Issue
- **Website**: [fairmapsforcalifornia.com](https://fairmapsforcalifornia.com/)

## Development Setup

```bash
git clone https://github.com/YOUR_ORG/fair-maps-for-california-open-code.git
cd fair-maps-for-california-open-code
pip install -e ".[dev]"
pytest tests/ -v
```

Replace `YOUR_ORG` with the actual GitHub org or username for this repository.
