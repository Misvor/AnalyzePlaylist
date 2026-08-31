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

## GUI

A local desktop web UI for the pipeline. Runs as a FastAPI app wrapped in a
pywebview window so it feels like a native program (youtube-dl-gui visual
style: action bar on top, list view in the middle, status bar at the bottom).
The window is loopback-only by design — **no auth, single-user**.

### Launch

The web entry point is `taste_pipeline.web`. The default mode opens a
pywebview desktop window; `--server` runs uvicorn in the foreground instead
(useful for headless tests or pointing your own browser at it):

```powershell
uv run python -m taste_pipeline.web          # default: desktop window via pywebview
uv run python -m taste_pipeline.web --server # server-only (browser / headless)
```

The bind address and port come from `config.toml` — see the `[web]` keys
(`web_host`, `web_port`) in `config.example.toml`. Defaults: `127.0.0.1:8741`.
Expose to a LAN only if you understand the implications; loopback-only is the
safe default.

### Pages

| Path | What it does |
| --- | --- |
| `/` (Dashboard) | Four run buttons: **Pass A — feed**, **Pass B — metadata**, **Pass C — download**, **Build index**. Each button POSTs to `/api/jobs` and a run-history table shows past runs with live progress via SSE + htmx (progress bars, status badges, error text). |
| `/triage` | Browse downloaded tracks from `inbox/`. Each row has an HTML5 `<audio controls>` element (range-request serving, seeking works) and four action buttons — **Keep** / **Review** / **Skip** / **Dislike** — that move the FLAC + sidecar into the matching `data_dir` subdir and update `downloads.stage`. |
| `/settings` | Editable form for every field in `Config`. Changes to **Taste profile** and **Pipeline** (thresholds, scalar limits, chunk sizes, model name) take effect immediately. Changes to **Storage** (`like_library_dir`, `data_dir`, `cookie_file`, `download_archive`) and **Network** (`web_host`, `web_port`) are persisted to `config.toml` but only take effect after a process restart — the page surfaces an "Unapplied path / network changes — restart required" banner listing every divergent field with its old and new value, so the user can see exactly what the next restart will pick up. The cookie file path is rendered, never its contents. |

### What the GUI does NOT do (scope guardrails)

- **No auto-scoring or clustering.** The pipeline itself does not score or
  cluster tracks; the GUI only exposes buttons for what already exists.
- **No authentication / multi-user.** Loopback single-user only.
- **No mobile / responsive design.** The app is a desktop window, not a
  phone-friendly web page.

### Dev commands

Run these from the `YTPlaylist/taste-pipeline` directory:

```powershell
# Fast test suite (skips tests marked `slow`; ~157 passed + 2 skipped).
uv run pytest -q

# Full suite including the slow CLAP embedding tests
# (downloads ~1.8GB of model weights the first time).
uv run pytest --runslow -q

# Lint (select=ALL). On Windows, pass --no-cache to dodge a uv rename error.
uv run ruff check src tests

# Format check. On Windows, pass --no-cache.
uv run ruff format --check src tests

# Strict type-check the web subsystem only. Pre-existing errors in committed
# non-web files are out of scope for this todo; the gate is 0/0/0 on web/.
uvx --offline --no-progress basedpyright src/taste_pipeline/web
```
