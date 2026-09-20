# AGENTS.md

## Project overview

This repo contains `piitag`, a Python package that wraps the Desert Ant Labs
Redact on-device PII detection pipeline.

- Package layout uses a `src/` layout under `src/piitag`.
- Runtime dependencies are managed with `uv`.
- Python requirement is 3.10 or newer.
- Model assets are not bundled in the distribution and are downloaded from the
  pinned Hugging Face revision at runtime or via the local test fixture
  workflow.
- Releases are driven by Conventional Commits and `python-semantic-release`.

## Working conventions

- Prefer `uv` commands over direct `python -m pip` usage.
- Keep changes reproducible; do not hand-edit the generated lockfile.
- Keep documentation in sync with behavior, especially `README.md`,
  `CHANGELOG.md`, and `docs/PLAN.md`.
- Do not add model weights or downloaded fixtures to git-tracked source files.
- Keep CI in a single workflow file at `.github/workflows/ci.yml`.

## Validation commands

```sh
uv run pytest
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run mypy src/
uv run python -m build
uv run twine check dist/*
```

## Commit and release expectations

Use Conventional Commits:

- `fix:` -> patch release
- `feat:` -> minor release
- `BREAKING CHANGE:` or `!` -> major release

## Model fixtures and test assets

The expected local asset directory is `tests/fixtures/model/`. It is for
fetched test assets and should not be committed.

## Notes for future agents

Keep the implementation aligned with `docs/PLAN.md`. Prefer minimal, surgical
edits over broad rewrites.
