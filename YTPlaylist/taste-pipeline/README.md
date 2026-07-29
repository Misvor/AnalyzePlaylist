# taste-pipeline

YouTube Music taste pipeline: downloads playlist audio, computes track
embeddings, clusters them into taste profiles, and serves the results over
HTTP.

## Requirements

- [uv](https://docs.astral.sh/uv/) — manages the Python 3.11 toolchain and all
  dependencies automatically.

## Install

```powershell
uv sync
```

This creates a local `.venv` (git-ignored) with every dependency from
`pyproject.toml` / `uv.lock`.

## Run tests

```powershell
uv run pytest            # fast tests only (tests marked `slow` are skipped)
uv run pytest --runslow  # include slow tests
```

## Lint

```powershell
uv run ruff check src tests
```
