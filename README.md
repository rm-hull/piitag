# piitag

Python port of Desert Ant Labs' Redact on-device PII detector.

The package is under active development. The implementation is being built in
stages described in [the implementation plan](docs/PLAN.md).

## Quick start

Install the package and one of the supported TFLite runtimes:

```sh
uv add piitag ai-edge-litert
```

The first detection downloads the pinned model assets into the Hugging Face
cache. You can also provide a directory containing the three model files:

```python
from piitag import Label, Options, Redact

redactor = Redact()
result = redactor.redaction("Email anna@example.com")
print(result.redacted_text)
print(result.restore(result.redacted_text))

only_email = redactor.redaction(
    "Email anna@example.com",
    Options(labels=frozenset({Label.EMAIL})),
)
```

The command-line interface accepts the same model cache by default:

```sh
piitag "Email anna@example.com"
piitag "Call +44 20 7946 0958" --labels PHONE --json
```

Use `--directory PATH` for an offline directory containing
`redact_tokenizer.bin`, `labels.json`, and `redact.tflite`.

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

## Release

Releases use Conventional Commits on `main` with
`python-semantic-release`. The release workflow updates the package version,
synchronizes `uv.lock`, updates [CHANGELOG.md](CHANGELOG.md), creates a tag
and GitHub release, and publishes the wheel and source archive to PyPI.

PyPI publishing uses Trusted Publishing. Configure the PyPI project publisher
for this repository, workflow, and the `pypi` environment before publishing.

- `fix:` creates a patch release.
- `feat:` creates a minor release.
- `BREAKING CHANGE:` or a `!` after the commit type creates a major release.

Preview the next release locally without changing files:

```sh
uv run semantic-release --noop version
```