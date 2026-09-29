"""CLAP audio embeddings: lazy model loading, explicit 10 s chunking, mean pooling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import override  # py311: typing.override exists only on 3.12+

from taste_pipeline.audio import chunk_audio, decode_audio

if TYPE_CHECKING:
    import torch
    from transformers import ClapModel, ClapProcessor
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    from taste_pipeline.config import Config

_EMBEDDING_DIM: Final = 512
# Chunks per CLAP audio forward pass: batching amortizes the GPU launch cost
# while keeping peak activation memory small (each 10 s chunk is ~1 s of mel).
_EMBED_BATCH_SIZE: Final[int] = 8
# The compute devices a config may request; "auto" resolves at call time.
_ALLOWED_DEVICES: Final[frozenset[str]] = frozenset({"auto", "cpu", "cuda", "mps"})

# CLAP weights bundled inside the installed package (populated by the
# maintainer; see src/taste_pipeline/models/clap/). The end user never
# selects or downloads this — get_model() always loads whatever is bundled;
# only the compute device is selectable.
_CLAP_WEIGHTS_DIR: Final[Path] = Path(__file__).resolve().parent / "models" / "clap"

# Lazy singleton cache keyed on (weights dir, resolved device): transformers/torch
# import and the model load happen on first use, never at module import, so
# importing this module (and test collection) stays cheap.
_MODEL_CACHE: dict[tuple[str, str], tuple[ClapModel, ClapProcessor]] = {}


def _resolve_device(requested: str) -> str:
    """Resolve a requested device to one that is actually usable here.

    ``auto`` prefers CUDA, then Apple MPS, then CPU. An explicit device that is
    unavailable (e.g. a stale ``cuda`` config on a CPU-only box) degrades to
    ``cpu`` rather than crashing; unknown names also degrade to ``cpu``.
    """
    import torch  # noqa: PLC0415  # lazy by design: keep torch out of module import

    if requested not in _ALLOWED_DEVICES or requested == "cpu":
        return "cpu"
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    if requested == "auto":
        return next((name for name in ("cuda", "mps") if available[name]), "cpu")
    if available[requested]:
        return requested
    return "cpu"


@dataclass(frozen=True, slots=True)
class EmbeddingError(Exception):
    """Raised when a track embedding is missing, non-finite, or has the wrong shape."""

    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        """Render the error with the offending path and the failure detail."""
        return f"failed to embed {self.path}: {self.detail}"


def get_model(device: str = "auto") -> tuple[ClapModel, ClapProcessor]:
    """Load (or return the cached) CLAP model and processor from the bundled weights directory.

    The weights live inside the installed package at
    ``src/taste_pipeline/models/clap/`` and are loaded strictly offline
    (``local_files_only=True``), so the app never contacts HuggingFace Hub.
    The model is loaded in float32 and moved to ``device`` (``auto`` resolves
    to CUDA, then MPS, then CPU); fp16/bf16 and ``model.half()`` are
    deliberately avoided so embeddings stay deterministic and comparable across
    runs. ``torch.manual_seed(0)`` is set at load time. Each distinct resolved
    device gets its own cached pair.

    Args:
        device: ``auto``, ``cpu``, ``cuda``, or ``mps``. Unavailable devices
            fall back to CPU (see :func:`_resolve_device`).

    Returns:
        The cached ``(model, processor)`` pair for the bundled weights.

    Raises:
        FileNotFoundError: ``pytorch_model.bin`` is not present under the
            bundled weights directory. Tells the user exactly what's missing
            so they can restore it.
    """
    resolved = _CLAP_WEIGHTS_DIR
    resolved_device = _resolve_device(device)
    cache_key = (str(resolved), resolved_device)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not (resolved / "pytorch_model.bin").is_file():
        message = (
            f"CLAP weights missing from the installation at {resolved}: "
            "expected pytorch_model.bin. The app ships with bundled weights; "
            "reinstall the app or restore that directory."
        )
        raise FileNotFoundError(message)
    import torch  # noqa: PLC0415  # lazy by design: keep torch/transformers out of module import
    from transformers import ClapModel, ClapProcessor  # noqa: PLC0415

    _ = torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]  # torch stub gap
    model = ClapModel.from_pretrained(  # pyright: ignore[reportUnknownMemberType]  # transformers stub gap
        resolved, torch_dtype=torch.float32, local_files_only=True
    ).eval()
    model = model.to(resolved_device)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]  # stub gap
    processor = ClapProcessor.from_pretrained(  # pyright: ignore[reportUnknownMemberType]  # stub gap
        resolved, local_files_only=True
    )
    pair = (model, processor)
    _MODEL_CACHE[cache_key] = pair
    return pair


def _embed_chunks(
    model: ClapModel,
    processor: ClapProcessor,
    chunks: list[npt.NDArray[np.float32]],
    sample_rate: int,
    device: str,
) -> npt.NDArray[np.float32]:
    """Embed audio chunks into a ``(len(chunks), 512)`` float32 array on CPU.

    Chunks are processed ``_EMBED_BATCH_SIZE`` at a time: the processor batches
    the group, its tensors are moved to ``device``, and one
    ``model.get_audio_features`` forward pass under ``torch.no_grad()`` yields
    the pooled vectors, which are returned on CPU (``.cpu()`` before numpy).
    """
    import torch  # noqa: PLC0415  # lazy by design: see get_model

    batches: list[npt.NDArray[np.float32]] = []
    for start in range(0, len(chunks), _EMBED_BATCH_SIZE):
        group = chunks[start : start + _EMBED_BATCH_SIZE]
        # transformers 5.x ProcessorMixin: audio goes in through the singular
        # ``audio`` kwarg (the 4.x ``audios`` no longer reaches the extractor);
        # a list of chunks is batched into a leading batch dim. The returned
        # BatchFeature is a str-keyed Tensor mapping. The stub's
        # ProcessingKwargs omits the audio kwargs, hence the per-kwarg ignores.
        inputs = cast(
            "dict[str, torch.Tensor]",
            processor(
                audio=group,
                sampling_rate=sample_rate,  # pyright: ignore[reportCallIssue]  # transformers stub gap
                return_tensors="pt",  # pyright: ignore[reportCallIssue]  # transformers stub gap
            ),
        )
        is_longer = inputs.get("is_longer")
        with torch.no_grad():
            # The stub's tuple | BaseModelOutputWithPooling union is the
            # can_return_tuple artifact; with the default return_dict the audio
            # tower always yields pooled output, so the casts are runtime-true.
            outputs = cast(
                "BaseModelOutputWithPooling",
                model.get_audio_features(  # pyright: ignore[reportUnknownMemberType]  # transformers stub gap
                    input_features=inputs["input_features"].to(device),
                    is_longer=None if is_longer is None else is_longer.to(device),
                ),
            )
        tensor = cast("torch.Tensor", outputs.pooler_output)
        batches.append(tensor.cpu().numpy())
    stacked = np.concatenate(batches, axis=0)
    return stacked.astype(np.float32)


def embed_track(audio_path: Path, config: Config) -> npt.NDArray[np.float32]:
    """Embed a track into a ``(512,)`` float32 L2-normalized CLAP vector.

    The track is decoded with ffmpeg, split into ``config.chunk_seconds`` windows
    (CLAP's native 10 s context; a trailing window shorter than
    ``config.min_chunk_seconds`` is dropped), embedded on ``config.device`` in
    batches of chunks per forward pass, then mean-pooled and L2-normalized.

    Args:
        audio_path: Audio file to embed (any format ffmpeg understands).
        config: Pipeline configuration supplying the sample rate, chunking, and
            compute device; the CLAP weights themselves are bundled with the
            app, not configurable.

    Returns:
        The track embedding: shape ``(512,)``, dtype float32, unit L2 norm.

    Raises:
        EmbeddingError: There are no embeddable chunks, or the pooled embedding
            contains NaN/Inf, has a zero norm, or has the wrong shape.
        DecodeError: The file cannot be decoded into PCM samples.
    """
    resolved_device = _resolve_device(config.device)
    model, processor = get_model(config.device)
    samples = decode_audio(audio_path, config.sample_rate)
    chunks = chunk_audio(samples, config.sample_rate, config.chunk_seconds, config.min_chunk_seconds)
    if not chunks:
        raise EmbeddingError(path=audio_path, detail="no audio chunks to embed")
    embeddings = _embed_chunks(model, processor, chunks, config.sample_rate, resolved_device)
    # numpy stubs type axis-reductions as Any; cast the (n_chunks, 512) ->
    # (512,) float32 mean back to the vector type it is at runtime.
    pooled = cast("npt.NDArray[np.float32]", embeddings.mean(axis=0))
    if pooled.shape != (_EMBEDDING_DIM,):
        raise EmbeddingError(
            path=audio_path, detail=f"expected embedding shape ({_EMBEDDING_DIM},), got {pooled.shape}"
        )
    if not np.all(np.isfinite(pooled)):
        raise EmbeddingError(path=audio_path, detail="embedding contains NaN or Inf")
    norm = float(np.linalg.norm(pooled))
    if norm == 0.0:
        raise EmbeddingError(path=audio_path, detail="embedding has zero L2 norm")
    return (pooled / norm).astype(np.float32)
