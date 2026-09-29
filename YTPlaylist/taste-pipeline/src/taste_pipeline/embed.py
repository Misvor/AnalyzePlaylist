"""CLAP audio embeddings: lazy model loading, explicit 10 s chunking, mean pooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import override  # py311: typing.override exists only on 3.12+

from taste_pipeline.audio import chunk_audio, decode_audio

if TYPE_CHECKING:
    from pathlib import Path

    import torch
    from transformers import ClapModel, ClapProcessor
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    from taste_pipeline.config import Config

_EMBEDDING_DIM: Final = 512

# Lazy singleton cache: transformers/torch import and the model download happen on
# first use, never at module import, so importing this module (and test collection)
# stays cheap.
_MODEL_CACHE: dict[str, tuple[ClapModel, ClapProcessor]] = {}


@dataclass(frozen=True, slots=True)
class EmbeddingError(Exception):
    """Raised when a track embedding is missing, non-finite, or has the wrong shape."""

    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        """Render the error with the offending path and the failure detail."""
        return f"failed to embed {self.path}: {self.detail}"


def get_model(model_path: str | Path) -> tuple[ClapModel, ClapProcessor]:
    """Load (or return the cached) CLAP model and processor from a local weights directory.

    The model is loaded in float32 on CPU and put in eval mode; fp16/bf16 and
    ``model.half()`` are deliberately avoided so embeddings stay deterministic
    and comparable across runs. ``torch.manual_seed(0)`` is set at load time.

    Args:
        model_path: Local filesystem path to the CLAP weights directory. Must
            contain ``pytorch_model.bin``, ``config.json``,
            ``preprocessor_config.json``, and ``tokenizer_config.json``
            (the standard ``transformers`` checkpoint layout, e.g. as
            downloaded from ``huggingface.co/laion/larger_clap_music_and_speech``).

    Returns:
        The cached ``(model, processor)`` pair for ``model_path``.

    Raises:
        FileNotFoundError: ``pytorch_model.bin`` is not present under
            ``model_path``. Tells the user exactly what's missing so they
            can re-download the weights.
    """
    resolved = Path(model_path).expanduser().resolve()
    cache_key = str(resolved)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not (resolved / "pytorch_model.bin").is_file():
        raise FileNotFoundError(
            f"CLAP weights not found at {resolved}: "
            "missing pytorch_model.bin. Place the four required files "
            "(pytorch_model.bin, config.json, preprocessor_config.json, "
            "tokenizer_config.json) under that directory, or set "
            "model_local_dir in config.toml to a populated weights dir."
        )
    import torch  # noqa: PLC0415  # lazy by design: keep torch/transformers out of module import
    from transformers import ClapModel, ClapProcessor  # noqa: PLC0415

    _ = torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]  # torch stub gap
    model = ClapModel.from_pretrained(  # pyright: ignore[reportUnknownMemberType]  # transformers stub gap
        resolved, torch_dtype=torch.float32
    ).eval()
    processor = ClapProcessor.from_pretrained(resolved)  # pyright: ignore[reportUnknownMemberType]  # stub gap
    pair = (model, processor)
    _MODEL_CACHE[cache_key] = pair
    return pair


def _embed_chunk(
    model: ClapModel,
    processor: ClapProcessor,
    chunk: npt.NDArray[np.float32],
    sample_rate: int,
) -> npt.NDArray[np.float32]:
    """Embed one audio chunk into a ``(512,)`` float32 vector via the CLAP audio tower."""
    import torch  # noqa: PLC0415  # lazy by design: see get_model

    # transformers 5.x ProcessorMixin: audio goes in through the singular
    # ``audio`` kwarg (the 4.x ``audios`` no longer reaches the extractor);
    # the returned BatchFeature is a str-keyed Tensor mapping. The stub's
    # ProcessingKwargs omits the audio kwargs, hence the per-kwarg ignores.
    inputs = cast(
        "dict[str, torch.Tensor]",
        processor(
            audio=chunk,
            sampling_rate=sample_rate,  # pyright: ignore[reportCallIssue]  # transformers stub gap
            return_tensors="pt",  # pyright: ignore[reportCallIssue]  # transformers stub gap
        ),
    )
    with torch.no_grad():
        # The stub's tuple | BaseModelOutputWithPooling union is the
        # can_return_tuple artifact; with the default return_dict the audio
        # tower always yields pooled output, so the casts are runtime-true.
        outputs = cast(
            "BaseModelOutputWithPooling",
            model.get_audio_features(  # pyright: ignore[reportUnknownMemberType]  # transformers stub gap
                input_features=inputs["input_features"],
                is_longer=inputs.get("is_longer"),
            ),
        )
    tensor = cast("torch.Tensor", outputs.pooler_output)
    vector: npt.NDArray[np.float32] = tensor.squeeze(0).cpu().numpy()
    return vector


def embed_track(audio_path: Path, config: Config) -> npt.NDArray[np.float32]:
    """Embed a track into a ``(512,)`` float32 L2-normalized CLAP vector.

    The track is decoded with ffmpeg, split into ``config.chunk_seconds`` windows
    (CLAP's native 10 s context; a trailing window shorter than
    ``config.min_chunk_seconds`` is dropped), embedded chunk by chunk, then
    mean-pooled and L2-normalized.

    Args:
        audio_path: Audio file to embed (any format ffmpeg understands).
        config: Pipeline configuration (model name, sample rate, chunking).

    Returns:
        The track embedding: shape ``(512,)``, dtype float32, unit L2 norm.

    Raises:
        EmbeddingError: There are no embeddable chunks, or the pooled embedding
            contains NaN/Inf, has a zero norm, or has the wrong shape.
        DecodeError: The file cannot be decoded into PCM samples.
    """
    model, processor = get_model(config.model_local_dir)
    samples = decode_audio(audio_path, config.sample_rate)
    chunks = chunk_audio(samples, config.sample_rate, config.chunk_seconds, config.min_chunk_seconds)
    if not chunks:
        raise EmbeddingError(path=audio_path, detail="no audio chunks to embed")
    embeddings = np.stack(
        [_embed_chunk(model, processor, chunk, config.sample_rate) for chunk in chunks], axis=0
    )
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
