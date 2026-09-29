"""Threshold calibration for taste classification.

Calibration reads the persisted like-library index built by
:func:`taste_pipeline.library.scan_library` and derives two
cosine-similarity quantiles from the pairwise similarity matrix:

- ``keep_threshold`` -- a high-percentile similarity (default 0.75). If a
  freshly downloaded track has cosine similarity ``>= keep_threshold``
  to the like-library distribution, it lands in ``keep/`` without user
  review.
- ``skip_threshold`` -- a low-percentile similarity (default 0.25). Tracks
  below this threshold land in ``skip/`` automatically.

The quantiles are written to ``<data_dir>/thresholds.json`` so the rest
of the pipeline (and the dashboard's index-status panel) can read them
without re-running calibration. The file also carries an ISO-8601
``computed_at`` timestamp so the UI can show "Calibrated 3 hours ago"
rather than just "Calibrated".

Public surface:

- :func:`compute_thresholds` -- pure numpy computation, no I/O. Returns a
  ``dict[str, float]``. Defensive default for degenerate inputs (< 2
  vectors).
- :func:`persist_thresholds` -- JSON write at ``data_dir/thresholds.json``.
- :func:`load_thresholds` -- inverse of :func:`persist_thresholds`; returns
  ``None`` when the file is absent so callers can branch on "not yet
  calibrated".
- :func:`run_calibration` -- the orchestrator: loads the already-built
  index via :func:`taste_pipeline.library.load_index` (it never walks or
  embeds the library), computes thresholds, persists, reports progress
  through the injected ``report`` callback. Returns the persisted
  thresholds dict so the caller (the ``calibrate`` job factory) can
  surface them to the UI.

This module imports :mod:`taste_pipeline.library` lazily inside
:func:`run_calibration` -- mirroring the ``job_factories`` pattern -- so
test code can ``monkeypatch.setattr`` the source module before
:func:`run_calibration` runs and observe the patched functions without
the calibrate module caching its own references.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from taste_pipeline.config import Config

_THRESHOLDS_FILE_NAME: str = "thresholds.json"
"""Filename for the persisted thresholds JSON under ``config.data_dir``."""

# Defensive defaults returned when the library has fewer than 2 tracks and
# pairwise similarity is undefined. These mirror sensible fallbacks:
# ``keep_threshold`` around the middle of the unit interval; ``skip_threshold``
# below it so the keep/skip split is asymmetric (more permissive on skip).
_DEFAULT_KEEP_THRESHOLD: float = 0.5
_DEFAULT_SKIP_THRESHOLD: float = 0.1

# Minimum vector count to compute a meaningful pairwise similarity quantile.
# With n < 2 there are no pairs and the distribution is undefined.
_MIN_VECTORS_FOR_PAIRWISE: int = 2


def compute_thresholds(
    vectors: npt.NDArray[np.float32],
    *,
    keep_quantile: float = 0.75,
    skip_quantile: float = 0.25,
) -> dict[str, float]:
    """Derive ``keep_threshold`` and ``skip_threshold`` from a vector matrix.

    Args:
        vectors: ``(n, d)`` float32 embedding matrix. ``d`` is the CLAP
            embedding dimension (512 for ``laion/larger_clap_music_and_speech``).
        keep_quantile: Quantile in ``[0, 1]`` for the high-percentile
            similarity that becomes ``keep_threshold``. Default ``0.75``.
        skip_quantile: Quantile in ``[0, 1]`` for the low-percentile
            similarity that becomes ``skip_threshold``. Default ``0.25``.

    Returns:
        ``{"keep_threshold": float, "skip_threshold": float}``. When
        ``n < 2`` (degenerate, no pairwise similarity is defined) the
        defensive defaults ``0.5`` / ``0.1`` are returned so callers can
        always read the two keys without conditional branches.
    """
    n = int(vectors.shape[0])  # pyright: ignore[reportAny] -- numpy stub gap (shape[0] returns int)
    if n < _MIN_VECTORS_FOR_PAIRWISE:
        return {"keep_threshold": _DEFAULT_KEEP_THRESHOLD, "skip_threshold": _DEFAULT_SKIP_THRESHOLD}
    # L2-normalize each row so dot product == cosine similarity. CLAP vectors
    # are already unit-norm at the embedder output, but the defensive
    # renormalize keeps this function correct for inputs that bypassed
    # :func:`taste_pipeline.embed.embed_track` (e.g. tests injecting raw vectors).
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)  # pyright: ignore[reportAny]
    norms = norms.astype(np.float32)  # pyright: ignore[reportAny]
    safe_norms = np.where(norms == np.float32(0.0), np.float32(1.0), norms)  # pyright: ignore[reportAny]
    normalized = (matrix / safe_norms).astype(np.float32, copy=False)
    similarity = normalized @ normalized.T
    # Diagonal is self-similarity (== 1.0 for unit vectors); exclude it
    # from the quantile computation so the keep-threshold isn't anchored at
    # the trivial maximum. ``np.fill_diagonal`` + ``triu_indices(n, k=1)``
    # both work; the upper-triangle form is the simpler one-liner.
    upper = similarity[np.triu_indices(n, k=1)]
    keep_threshold = float(np.quantile(upper, keep_quantile))
    skip_threshold = float(np.quantile(upper, skip_quantile))
    return {"keep_threshold": keep_threshold, "skip_threshold": skip_threshold}


def persist_thresholds(
    data_dir: Path,
    thresholds: dict[str, float],
    *,
    sample_count: int | None = None,
) -> Path:
    """Write the thresholds dict to ``<data_dir>/thresholds.json`` and return the path.

    The JSON payload includes ``computed_at`` (ISO-8601 UTC) so the UI
    can render "Calibrated N hours ago" without recomputing from file
    mtime. When ``sample_count`` is provided it is recorded alongside the
    thresholds so the UI can warn that a small sample is statistically
    noisy. Atomic write is NOT used here because calibration is rare and
    the partial-write failure mode is recoverable on the next run.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = dict(thresholds)
    payload["computed_at"] = datetime.now(UTC).isoformat()
    if sample_count is not None:
        payload["sample_count"] = sample_count
    out_path = data_dir / _THRESHOLDS_FILE_NAME
    _ = out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def load_thresholds(data_dir: Path) -> dict[str, float] | None:
    """Load ``<data_dir>/thresholds.json``; return ``None`` when the file is absent.

    A malformed file is treated as missing (returns ``None``) so the UI
    does not have to distinguish "never calibrated" from "calibration
    crashed mid-write" -- both states surface as "not calibrated".
    """
    path = data_dir / _THRESHOLDS_FILE_NAME
    if not path.is_file():
        return None
    try:
        loaded = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    typed = cast("dict[str, object]", loaded)
    keep = typed.get("keep_threshold")
    skip = typed.get("skip_threshold")
    if not isinstance(keep, (int, float)) or not isinstance(skip, (int, float)):
        return None
    return {"keep_threshold": float(keep), "skip_threshold": float(skip)}


def run_calibration(
    config: Config,
    report: Callable[[float, str], None],
) -> dict[str, float]:
    """Load the persisted index, derive thresholds, and persist them.

    The ``report`` callback follows the same ``(progress, message)``
    contract every other pipeline pass uses; the runner turns those into
    SSE events for the dashboard's progress bar.

    Args:
        config: Pipeline configuration. The index is read from
            ``config.data_dir``; ``config.data_dir`` is where
            thresholds.json is written. ``config.like_library_dir`` is
            NOT walked -- run the index job first.
        report: ``Callable[[float, str], None]`` -- the runner's
            progress sink. Called at 0.0 / 0.3 / 0.9 / 1.0 with
            human-readable status messages.

    Returns:
        The persisted thresholds dict ``{"keep_threshold", "skip_threshold"}``.
    """
    # Lazy imports mirror the ``job_factories`` pattern: tests monkey-patch
    # ``taste_pipeline.library.load_index`` before submitting the job and rely
    # on module-attribute lookup at call time.
    from taste_pipeline import library  # noqa: PLC0415 -- lazy: monkey-patch works

    report(0.0, "loading index")
    index = library.load_index(config)
    count = index.count()
    report(0.3, f"indexed {count} tracks")
    vectors = index.vectors()
    thresholds = compute_thresholds(vectors)
    report(0.9, "persisting thresholds")
    _ = persist_thresholds(config.data_dir, thresholds, sample_count=count)
    report(1.0, "calibration complete")
    return thresholds
