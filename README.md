# piitag

Python port of Desert Ant Labs' Redact on-device PII detector.

The package is under active development. The implementation is being built in
stages described in [the implementation plan](docs/PLAN.md).

## Development setup

Requirements:

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)

Create an isolated environment and install the development dependencies:

```sh
uv venv
source .venv/bin/activate
uv sync --extra dev --extra litert
```

Use `--extra tensorflow` instead of `--extra litert` when TensorFlow is the
preferred TFLite runtime.

## Build, test, and check

```sh
uv run pytest
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run mypy src/
uv run python -m build
uv run twine check dist/*
```

Model weights and taxonomy assets are not bundled with the package. See
[NOTICE.md](NOTICE.md) for the applicable model license.