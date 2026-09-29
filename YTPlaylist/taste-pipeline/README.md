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

## CLAP weights

The CLAP audio model (~776 MB) is bundled inside the package at
`src/taste_pipeline/models/clap/` and loaded locally with `local_files_only=True`,
so the app never contacts HuggingFace Hub. There is deliberately no config key,
CLI flag, or env var to change the model — `get_model()` always loads whatever
ships with the installation.

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
   every track through the CLAP model, and writes the manifest +
   `<data_dir>/index/` artifacts. Required before Calibrate or Check.
   The build is **resumable** — see "Resumable / pausable index build"
   below.
2. **`Calibrate weights`** on the Dashboard. Computes pairwise-cosine
   quantiles from the **persisted index** and persists
   `<data_dir>/thresholds.json` with `keep_threshold` and
   `skip_threshold` (plus `sample_count`). After this, every track on
   the **Check** page gets a verdict against the calibrated thresholds.
3. Optional — export a YouTube `cookies.txt` and set `cookie_file`
   in `config.toml`. Required only for the Check-URL flow against
   YouTube (which does bot-detection on anonymous fetches).

### Pages

| Path | What it does |
| --- | --- |
| `/` (Dashboard) | Five run buttons: **Pass A — feed**, **Pass B — metadata**, **Pass C — download**, **Build index**, **Calibrate weights**. Each button POSTs to `/api/jobs` and a run-history table shows past runs with live progress via SSE + htmx (progress bars, status badges, error text). The Index Status panel shows library size + last-scan timestamp + the active keep/skip thresholds (or "(not calibrated)") + the configured compute device. |
| `/triage` | Browse downloaded tracks from `inbox/`. Each row has an HTML5 `<audio controls>` element (range-request serving, seeking works) and four action buttons — **Keep** / **Review** / **Skip** / **Dislike** — that move the FLAC + sidecar into the matching `data_dir` subdir and update `downloads.stage`. |
| `/check` | Single-track taste check against the like-library. Two input modes: (1) drag-and-drop or file-pick a local audio file (sent as base64-in-JSON to avoid the python-multipart dep), or (2) paste a YouTube URL which kicks off a `check_url` job that runs fetch → metadata → download → embed → score. The result panel shows the verdict (`Matches your taste` / `Uncertain` / `Doesn't match`), the cosine score, and the top-5 nearest library neighbours. Requires **Build Index** (and ideally **Calibrate**) to have run first. |
| `/index` | Live progress for the like-library index scan: the **current song** being embedded, the **upcoming queue** (up to 10 paths) with a **remaining** count, `processed / total`, **elapsed** time, **ETA**, a progress bar, and a scrolling log. **Build Index** starts a new scan; **Pause** stops the scan at the next checkpoint boundary while keeping every embedding so far; **Resume** is a fresh Build Index click that picks up where the previous run left off. The **Batch** input caps how many NEW files this run embeds (empty = the whole library). The status panel (indexed / pending / total, last scan, like-library size, compute device, keep/skip thresholds) refreshes when the scan reaches a terminal state. |
| `/settings` | Editable form for every field in `Config`. **Browse...** buttons next to every path field open the host OS native file/directory picker (pywebview desktop mode) or the browser file picker (`--server` mode). Changes to **Taste profile** and **Pipeline** (thresholds, scalar limits, chunk sizes, **compute device**) take effect immediately. Changes to **Storage** (`like_library_dir`, `data_dir`, `cookie_file`, `download_archive`) and **Network** (`web_host`, `web_port`) are persisted to `config.toml` but only take effect after a process restart — the page surfaces an "Unapplied path / network changes — restart required" banner listing every divergent field with its old and new value, so the user can see exactly what the next restart will pick up. The cookie file path is rendered, never its contents. A dirty-form guard prompts before navigation while there are unsaved changes. |

### Resumable / pausable index build

The like-library index is persisted incrementally so a long scan never
has to start from scratch:

- **Incremental checkpoints.** After every batch of new embeddings
  (default: every 200 embeds or every 60 s, whichever comes first) the
  runner writes `library_vectors.npy.tmp` and `library_manifest.json.tmp`
  to disk and atomically `Path.replace`s them onto the live files.
  A process crash, a power loss, or a `Pause` click loses at most one
  batch — the next run picks up at the last checkpoint.
- **Partial index is always usable.** `Calibrate weights` and the
  `/check` page now read the persisted index via `load_index(...)`
  instead of re-scanning the library. A check against an un-indexed
  library returns "Run Build Index first" instead of kicking off a
  six-hour re-embed.
- **Pause / Resume.** While an index job is in flight, the Index page
  shows a **Pause** button. Clicking it flips the job to `paused` at
  the next checkpoint boundary. **Resume** is a fresh **Build Index**
  click — the runner skips every track already in the manifest and only
  embeds the missing ones, so it costs as much time as the songs you
  added since the last run.
- **Per-run batch size.** The **Batch** input on the Index page caps
  the number of NEW files a single run will embed. Leave it empty to
  index the whole remaining library in one go; set it to e.g. 200 to
  chip away at the work in nightly-sized chunks. The cap is a ceiling,
  not a target — if only 50 tracks are missing the run stops at 50.

### GPU support

Embeddings run on the compute device configured in
`config.device` (Settings → Pipeline → `device`, or `device = "auto"`
in `config.toml`). The accepted values are:

| Value | Behavior |
| --- | --- |
| `auto` (default) | `cuda` if a CUDA GPU is reachable, else `mps` (Apple Silicon), else `cpu`. Safe across machines. |
| `cuda` | Force the CUDA device. Requires an NVIDIA GPU + the CUDA-enabled PyTorch wheel (see below). |
| `mps` | Force Apple Silicon Metal. Requires PyTorch built with MPS support. |
| `cpu` | Force CPU. Always available, slowest. |

Forward passes are batched (8 chunks per forward pass) for GPU
throughput, with model weights and inputs moved to the resolved device
and outputs moved back to CPU before stacking the numpy matrix. The
model is cached per `(weights_dir, device)` so a second embed on the
same device is a cache hit.

#### CUDA PyTorch wheel

`pyproject.toml` pins `torch` to the CUDA 12.8 index, so `uv sync`
installs the GPU build directly — no CPU-wheel override step. This covers
Blackwell (sm_120) cards such as the RTX 5080 on a recent driver. CUDA
wheels still run on CPU when no NVIDIA GPU is present, so the project
works on machines without one.

To target a different CUDA tag, edit the `pytorch-cu128` index URL in
`pyproject.toml` (e.g. `cu126`) and force a clean reinstall:

```powershell
uv sync --reinstall-package torch
```

Without a CUDA wheel, `device = "auto"` and `device = "cuda"` both fall
back to `cpu` (and the Index page status panel still shows the configured
value). Set `device = "cpu"` explicitly to be sure no surprise GPU
selection happens.

#### Proxy (blocked networks)

If `youtube.com` does not resolve (`getaddrinfo failed`) the pipeline cannot
reach YouTube without a proxy/VPN. Set `proxy` in `config.toml` — it is passed
to every yt-dlp call (feed, metadata, download, and the Check-URL parse):

```toml
proxy = "http://127.0.0.1:56551"   # or socks5://127.0.0.1:1080
```

Make sure the proxy client is running and listening on that port (yt-dlp
reports `Unable to connect to proxy` when nothing is bound). yt-dlp also honors
the `HTTP_PROXY` / `HTTPS_PROXY` environment variables if you would rather not
set it here.

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
# Fast test suite (skips tests marked `slow`; 235 passed + 2 skipped).
uv run pytest -q

# Full suite including the slow CLAP embedding tests. The CLAP weights are
# bundled with the app (src/taste_pipeline/models/clap/), so these run fully
# offline -- there is no download and no model-selection setting.
uv run pytest --runslow -q

# Lint (select=ALL). On Windows, pass --no-cache to dodge a uv rename error.
uv run ruff check --no-cache src tests

# Format check. On Windows, pass --no-cache.
uv run ruff format --no-cache --check src tests

# Strict type-check the web subsystem only. Pre-existing errors in committed
# non-web files are out of scope; the gate is 0/0/0 on web/.
uvx --offline --no-progress basedpyright src/taste_pipeline/web
```
