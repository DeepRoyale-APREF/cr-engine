# Clash Royale Engine - Agent Guide

This repository contains a high-performance Clash Royale Arena 1 simulation engine for Reinforcement Learning (RL) training.
It is a Python project using `setuptools` for build and `pytest` for testing.

## 1. Build, Lint, and Test

### Installation
The project uses `pip` with optional dependencies.
```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install with RL dependencies
pip install -e ".[rl]"
```

### Testing
Tests are located in `tests/` and run with `pytest`.
```bash
# Run all tests
pytest

# Run tests with short tracebacks (configured in pyproject.toml)
pytest -v

# Run a specific test file
pytest tests/test_engine.py

# Run a specific test class
pytest tests/test_engine.py::TestGameEngine

# Run a single test method
pytest tests/test_engine.py::TestGameEngine::test_initialization

# Run non-slow tests (if markers are used to exclude slow ones)
pytest -m "not slow"
```

### Linting and Formatting
The project enforces strict code style using `pyrefly`, `ruff`, and `mypy`.
```bash
# Format code (auto-fix)
ruff check --fix .

# Type checking (strict)
mypy .
```

Configuration is found in `pyproject.toml`.
- **Line Length**: 100 characters.
- **Python Version**: Target 3.12.

## 2. Code Style & Conventions

### General
- **Language**: Python 3.10+ (Target 3.12).
- **Paradigms**: Object-Oriented Programming (OOP) with strong typing.
- **Docstrings**: Required for all public classes and methods. Use Google-style docstrings (Parameters, Returns sections).

### Imports
- Use `from __future__ import annotations` at the top of every file.
- Imports should be sorted by `isort` (standard library -> third party -> local).
- Use absolute imports for local modules (e.g., `from clash_royale_engine.core.state import State`).

### Naming
- **Classes**: `CamelCase` (e.g., `ClashRoyaleEngine`, `CombatSystem`).
- **Functions/Methods**: `snake_case` (e.g., `get_state`, `_tick_one_frame`).
- **Variables**: `snake_case`.
- **Constants**: `UPPER_CASE` (e.g., `DEFAULT_FPS`).
- **Private Members**: Prefix with `_` (e.g., `_apply_action`).

### Typing
- **Strict Typing**: All function signatures must have type hints.
- **Return Types**: Must be explicitly declared (`-> None`, `-> State`).
- Use `typing.Optional`, `typing.List`, `typing.Tuple`, `typing.Dict` or standard collection types if on Python 3.9+.
- Resolve circular imports using `if typing.TYPE_CHECKING:` blocks for type-only imports.

### Error Handling
- Use custom exception classes for domain-specific errors (e.g., `InvalidActionError`).
- Fail fast and explicitly.

### File Structure
- **Core Logic**: `clash_royale_engine/core/` (Engine, State, Arena).
- **Entities**: `clash_royale_engine/entities/` (Troops, Buildings).
- **Systems**: `clash_royale_engine/systems/` (Physics, Combat, Elixir).
- **Tests**: `tests/` (Mirroring structure where possible).

## 3. Testing Guidelines

- **Framework**: `pytest`.
- **Structure**: Group related tests in classes (e.g., `class TestCombatDamage:`).
- **Fixtures**: Use `pytest.fixture` for common setup (e.g., creating a fresh engine instance).
- **Coverage**: Aim for high coverage of core logic.
- **Performance**: Mark slow tests with `@pytest.mark.slow`.

## 4. Agent Behavior
- **Refactoring**: When modifying core engine logic, always run existing tests to ensure no regression.
- **New Features**: Add corresponding tests in `tests/`.
- **Safety**: Do not commit changes that fail `mypy` or `pytest`.
