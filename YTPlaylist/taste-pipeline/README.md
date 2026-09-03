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
# Required once — copy the example config and point the paths at your setup
cp config.example.toml config.toml
# Edit config.toml: set like_library_dir / data_dir / cookie_file

# Default: desktop window via pywebview (uses config.web_host:config.web_port)
uv run python -m taste_pipeline.web --config config.toml

# Server-only (browser / headless). Open http://127.0.0.1:8741 (or your port) in a browser.
uv run python -m taste_pipeline.web --config config.toml --server
```

The bind address and port come from `config.toml` — see the `[web]` keys
(`web_host`, `web_port`) in `config.example.toml`. Defaults: `127.0.0.1:8741`.
Expose to a LAN only if you understand the implications; loopback-only is the
safe default.

### First-time workflow (one-time setup)

After launching, work through these in order — the later buttons depend
on the earlier ones:

1. **`Build Index`** on the Dashboard. Scans `like_library_dir`, embeds
   every track through the CLAP model, writes the manifest +
   `<data_dir>/index/` artifacts. Required before Calibrate or Check.
2. **`Calibrate weights`** on the Dashboard. Scans the like-library
   again, computes pairwise-cosine quantiles, and persists
   `<data_dir>/thresholds.json` with `keep_threshold` and
   `skip_threshold`. After this, every track on the **Check** page gets
   a verdict against the calibrated thresholds.
3. Optional — export a YouTube `cookies.txt` and set `cookie_file`
   in `config.toml`. Required only for the Check-URL flow against
   YouTube (which does bot-detection on anonymous fetches).

### Pages

| Path | What it does |
| --- | --- |
| `/` (Dashboard) | Five run buttons: **Pass A — feed**, **Pass B — metadata**, **Pass C — download**, **Build index**, **Calibrate weights**. Each button POSTs to `/api/jobs` and a run-history table shows past runs with live progress via SSE + htmx (progress bars, status badges, error text). The Index Status panel shows library size + last-scan timestamp + the active keep/skip thresholds (or "(not calibrated)"). |
| `/triage` | Browse downloaded tracks from `inbox/`. Each row has an HTML5 `<audio controls>` element (range-request serving, seeking works) and four action buttons — **Keep** / **Review** / **Skip** / **Dislike** — that move the FLAC + sidecar into the matching `data_dir` subdir and update `downloads.stage`. |
| `/check` | Single-track taste check against the like-library. Two input modes: (1) drag-and-drop or file-pick a local audio file (sent as base64-in-JSON to avoid the python-multipart dep), or (2) paste a YouTube URL which kicks off a `check_url` job that runs fetch → metadata → download → embed → score. The result panel shows the verdict (`Matches your taste` / `Uncertain` / `Doesn't match`), the cosine score, and the top-5 nearest library neighbours. Requires **Build Index** (and ideally **Calibrate**) to have run first. |
| `/settings` | Editable form for every field in `Config`. **Browse...** buttons next to every path field open the host OS native file/directory picker (pywebview desktop mode) or the browser file picker (`--server` mode). Changes to **Taste profile** and **Pipeline** (thresholds, scalar limits, chunk sizes, model name) take effect immediately. Changes to **Storage** (`like_library_dir`, `data_dir`, `cookie_file`, `download_archive`) and **Network** (`web_host`, `web_port`) are persisted to `config.toml` but only take effect after a process restart — the page surfaces an "Unapplied path / network changes — restart required" banner listing every divergent field with its old and new value, so the user can see exactly what the next restart will pick up. The cookie file path is rendered, never its contents. A dirty-form guard prompts before navigation while there are unsaved changes. |

### What the GUI does NOT do (scope guardrails)

- **No auto-classification of new downloads.** Calibration computes the
  keep/skip thresholds from the like-library; the `/check` page is the
  one-off track scoring tool. The pipeline does not yet classify tracks
  during Pass C automatically against those thresholds.
- **No authentication / multi-user.** Loopback single-user only.
- **No mobile / responsive design.** The app is a desktop window, not a
  phone-friendly web page.

### Dev commands

Run these from the `YTPlaylist/taste-pipeline` directory:

```powershell
# Fast test suite (skips tests marked `slow`; 232 passed + 2 skipped).
uv run pytest -q

# Full suite including the slow CLAP embedding tests
# (downloads ~1.8GB of model weights the first time).
uv run pytest --runslow -q

# Lint (select=ALL). On Windows, pass --no-cache to dodge a uv rename error.
uv run ruff check --no-cache src tests

# Format check. On Windows, pass --no-cache.
uv run ruff format --no-cache --check src tests

# Strict type-check the web subsystem only. Pre-existing errors in committed
# non-web files are out of scope; the gate is 0/0/0 on web/.
uvx --offline --no-progress basedpyright src/taste_pipeline/web
```
