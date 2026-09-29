"""Track similarity check: compare a single track embedding against the like-library index.

The check is a read-only, non-mutating operation over the persisted library
embedding matrix produced by :mod:`taste_pipeline.library`. It scores a
``(512,)`` query vector against every indexed track via cosine similarity,
sorts the top-K matches, and classifies the verdict against the user's
calibrated thresholds (or returns ``uncertain`` when calibration has not
been run).

Public surface:

- :class:`CheckResult` -- frozen dataclass with ``verdict``, ``score``,
  ``top_k``, threshold echoes, and a human-readable summary.
- :func:`check_track_against_library` -- pure numpy function; takes a
  query vector + library matrix + ids + thresholds; returns a
  :class:`CheckResult`. Does not touch disk.
- :func:`check_audio_file` -- thin I/O wrapper: embeds the query audio
  file, loads the persisted index and thresholds from
  ``<data_dir>``, then calls the pure function. Mirrors
  :mod:`taste_pipeline.calibrate`'s lazy-import pattern so tests can
  monkey-patch ``embed.embed_track`` and ``library.load_index``
  without caching references at module load time. It never builds the
  index as a side effect -- an un-indexed library returns ``no_library``.

The "no calibration" case is handled explicitly: ``thresholds is None``
yields ``verdict="uncertain"`` with the top-K matches still populated so
the UI can still show the closest library neighbors. This is the only
case the pure function branches on threshold presence; it never raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pathlib import Path

    from taste_pipeline.config import Config

Verdict = Literal["matches", "uncertain", "doesn't_match", "no_library", "error"]
"""One of the five verdict states the check endpoint can return.

- ``matches`` -- score >= keep_threshold (the track fits the user's taste)
- ``uncertain`` -- skip_threshold <= score < keep_threshold, OR no
  calibration is on disk
- ``doesn't_match`` -- score < skip_threshold (the track is unlike the user's taste)
- ``no_library`` -- the like-library index is empty (run ``Build Index`` first)
- ``error`` -- the check could not be performed (raised exception)
"""

_EMBED_DIM: int = 512
_DEFAULT_TOP_K: int = 5


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The outcome of comparing one track embedding against the library index.

    Attributes:
        verdict: Classification of the match. ``"matches"`` means the
            track fits the user's taste, ``"doesn't_match"`` means it is
            unlike their taste, ``"uncertain"`` covers the in-between
            band (and the "not calibrated" case). ``"no_library"``
            signals an empty library index; ``"error"`` is reserved for
            unexpected failures.
        score: The maximum cosine similarity between the query track and
            the library matrix, in ``[0, 1]`` (clamped at zero; L2-normalized
            vectors cannot exceed 1.0, but defensive clamp protects against
            floating-point drift).
        top_k: Top-K (similarity, library_track_id) pairs, sorted by
            similarity descending. ``library_track_id`` is the
            ``rel_path`` of the indexed track (the manifest key in
            :mod:`taste_pipeline.library`).
        threshold_keep: The ``keep_threshold`` from calibration, or
            ``None`` when calibration has not been run.
        threshold_skip: The ``skip_threshold`` from calibration, or
            ``None`` when calibration has not been run.
        human_readable: A pre-formatted one-line summary, e.g.
            ``"Matches your taste (0.82 >= 0.75)"``. The check page
            surfaces this verbatim so the user does not need to read the
            verdict enum.
    """

    verdict: Verdict
    score: float
    top_k: list[tuple[float, str]] = field(default_factory=list)
    threshold_keep: float | None = None
    threshold_skip: float | None = None
    human_readable: str = ""


def _to_unit_vectors(matrix: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """L2-normalize each row of ``matrix``; rows with zero norm become unit vectors.

    CLAP embeddings are already unit-norm at the embedder output, but the
    defensive renormalize keeps this function correct for inputs that
    bypassed :func:`taste_pipeline.embed.embed_track` (e.g. tests injecting
    raw vectors or matrices with a numeric artifact).
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)  # pyright: ignore[reportAny]
    safe = np.where(norms == np.float32(0.0), np.float32(1.0), norms)  # pyright: ignore[reportAny]
    return (matrix / safe).astype(np.float32, copy=False)


def _pair_from_index(
    cosine: npt.NDArray[np.float32],
    library_ids: list[str],
    idx: np.intp,
) -> tuple[float, str]:
    """Resolve one numpy scalar to a ``(similarity, library_track_id)`` pair.

    Extracted from :func:`check_track_against_library` so the inline list
    comprehension does not blow past basedpyright's ``reportAny`` for
    ``int(idx)`` (the ``npt.NDArray`` element type narrows only after
    the cast, and inlining the cast twice ran past the line-length cap).
    """
    int_idx = cast("int", int(idx))
    return (float(cosine[int_idx]), library_ids[int_idx])


def _verdict_and_summary(
    score: float,
    thresholds: dict[str, float] | None,
) -> tuple[Verdict, float | None, float | None, str]:
    """Pick the verdict + summary from a score and the (possibly absent) thresholds.

    Returns:
        ``(verdict, threshold_keep, threshold_skip, human_readable)``. The
        threshold echoes are propagated so the caller can include them in
        the :class:`CheckResult` without re-reading the dict.
    """
    if thresholds is None:
        return "uncertain", None, None, f"Uncertain (no calibration: score {score:.2f})"
    keep = thresholds["keep_threshold"]
    skip = thresholds["skip_threshold"]
    if score >= keep:
        return "matches", keep, skip, f"Matches your taste ({score:.2f} >= {keep:.2f})"
    if score < skip:
        return "doesn't_match", keep, skip, f"Doesn't match ({score:.2f} < {skip:.2f})"
    return "uncertain", keep, skip, f"Uncertain ({score:.2f} between {skip:.2f} and {keep:.2f})"


def check_track_against_library(
    track_embedding: np.ndarray,
    library_vectors: np.ndarray,
    library_ids: list[str],
    thresholds: dict[str, float] | None,
    top_k: int = _DEFAULT_TOP_K,
) -> CheckResult:
    """Compare one track embedding against the library matrix and return a :class:`CheckResult`.

    Args:
        track_embedding: ``(512,)`` query vector. Coerced to float32 and
            L2-normalized defensively.
        library_vectors: ``(n, 512)`` library matrix (the output of
            :meth:`taste_pipeline.library.LibraryIndex.vectors`).
        library_ids: ``list[str]`` of library track ids aligned with the
            rows of ``library_vectors`` (the output of
            :meth:`LibraryIndex.manifest` keys, in the same order as the
            matrix rows). The library module documents that the manifest's
            ``vector_id`` aligns with the matrix row index; we use the
            input ordering here for ``top_k`` lookups.
        thresholds: The calibration thresholds dict
            (``{"keep_threshold", "skip_threshold"}``), or ``None`` when
            the user has not run calibration.
        top_k: How many top matches to surface. Default ``5``.

    Returns:
        A :class:`CheckResult`. The function never raises -- an empty
        library yields ``verdict="no_library"`` and the top-K matches
        field is an empty list.
    """
    query = cast("npt.NDArray[np.float32]", np.asarray(track_embedding, dtype=np.float32).reshape(-1))
    matrix = cast("npt.NDArray[np.float32]", np.asarray(library_vectors, dtype=np.float32))
    if matrix.shape[0] == 0:
        return CheckResult(
            verdict="no_library",
            score=0.0,
            human_readable="Library not indexed. Run Build Index first.",
        )

    # Cosine similarity = dot product of unit vectors. The query and every
    # row are L2-normalized defensively (CLAP vectors are already unit-norm
    # at the embedder, but a 0-norm defensive clamp avoids NaN propagation).
    unit_query = _to_unit_vectors(query.reshape(1, -1))[0]
    unit_matrix = _to_unit_vectors(matrix)
    cosine = cast("npt.NDArray[np.float32]", unit_query @ unit_matrix.T)
    # Clamp to [0, 1] -- unit-vector cosine cannot exceed 1.0; floating-point
    # drift can push it to 1.0 + epsilon, which would render as 1.00 instead
    # of the true <1.0 on the UI.
    score = float(np.clip(cosine.max(), np.float32(0.0), np.float32(1.0))) if cosine.size else 0.0

    # Top-K: sort the cosine vector, take the last ``top_k`` indices (highest
    # values), reverse to descending. ``min(top_k, n)`` so the user with a
    # tiny library still gets a complete top-K instead of a partial slice.
    k = min(top_k, int(matrix.shape[0]))
    top_indices = cast("npt.NDArray[np.intp]", np.argsort(cosine)[-k:][::-1])
    top_k_pairs: list[tuple[float, str]] = [_pair_from_index(cosine, library_ids, idx) for idx in top_indices]

    verdict, keep, skip, summary = _verdict_and_summary(score, thresholds)
    return CheckResult(
        verdict=verdict,
        score=score,
        top_k=top_k_pairs,
        threshold_keep=keep,
        threshold_skip=skip,
        human_readable=summary,
    )


def check_audio_file(audio_path: Path, config: Config) -> CheckResult:
    """Embed ``audio_path``, load the persisted index, compare, return a :class:`CheckResult`.

    This is the I/O wrapper: it embeds the single query track (the library
    is never re-embedded), loads the persisted library index via
    :func:`taste_pipeline.library.load_index`, loads the persisted
    calibration thresholds (or ``None``), then delegates to
    :func:`check_track_against_library`.

    The library ids are taken in row (``vector_id``) order via
    :meth:`taste_pipeline.library.LibraryIndex.ids` -- the manifest is not
    re-sorted, because incremental appends need not be globally sorted and
    row order is what aligns the ids with the vector matrix.

    The pipeline modules are imported lazily inside this function so
    tests can ``monkeypatch.setattr`` the source module attributes
    (``taste_pipeline.embed.embed_track``,
    ``taste_pipeline.library.load_index``,
    ``taste_pipeline.calibrate.load_thresholds``) and observe the patched
    call sites. The lazy-import pattern mirrors
    :func:`taste_pipeline.calibrate.run_calibration` so the existing test
    harness (and the documentation in :mod:`taste_pipeline.calibrate`)
    applies here too.

    Args:
        audio_path: Audio file to check (any format the CLAP embedder
            understands via ffmpeg).
        config: Pipeline configuration (``data_dir`` carries the persisted
            index and thresholds).

    Returns:
        A :class:`CheckResult`. Never raises -- embedding failures are
        surfaced via the embedder's exception types, which propagate to
        the caller (the route handler converts them to a 500). An empty or
        missing index surfaces as ``verdict="no_library"`` (run the index
        job first; this wrapper does NOT build it as a side effect).
    """
    # Lazy imports mirror the calibrate.py pattern: tests monkey-patch the
    # source modules before calling this function and rely on attribute
    # lookup at call time. See taste_pipeline/calibrate.py for the same
    # contract, documented in the calibrate module's docstring.
    from taste_pipeline import calibrate, embed, library  # noqa: PLC0415 -- lazy: monkey-patch works

    embedding = embed.embed_track(audio_path, config)
    index = library.load_index(config)
    thresholds = calibrate.load_thresholds(config.data_dir)
    return check_track_against_library(embedding, index.vectors(), index.ids(), thresholds)
