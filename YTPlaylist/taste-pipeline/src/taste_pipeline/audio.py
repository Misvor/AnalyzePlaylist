"""Audio decoding (via an ffmpeg subprocess) and fixed-window chunking."""

import shutil
import subprocess  # ffmpeg is invoked by design; argv list form, never shell=True
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from typing_extensions import override  # py311: typing.override exists only on 3.12+

_FFMPEG_BINARY: Final = "ffmpeg"


@dataclass(frozen=True, slots=True)
class DecodeError(Exception):
    """Raised when ffmpeg cannot decode an audio file into PCM samples."""

    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        """Render the error with the offending path and ffmpeg's stderr detail."""
        return f"failed to decode {self.path}: {self.detail}"


def decode_audio(path: Path, sample_rate: int) -> npt.NDArray[np.float32]:
    """Decode an audio file to mono float32 PCM at ``sample_rate`` using ffmpeg.

    Args:
        path: Audio file to decode (any container/codec ffmpeg understands).
        sample_rate: Target sample rate in Hz (CLAP expects 48000).

    Returns:
        One-dimensional float32 array of shape ``(n_samples,)``.

    Raises:
        DecodeError: ffmpeg is missing from PATH, exits nonzero, or yields no samples.
    """
    ffmpeg = shutil.which(_FFMPEG_BINARY)
    if ffmpeg is None:
        raise DecodeError(path=path, detail="ffmpeg executable not found on PATH")
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    # argv list (no shell) with a fully resolved executable: no injection surface.
    result = subprocess.run(command, capture_output=True, check=False)  # noqa: S603
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip() or f"exit code {result.returncode}"
        raise DecodeError(path=path, detail=detail)
    samples: npt.NDArray[np.float32] = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size == 0:
        raise DecodeError(path=path, detail="ffmpeg produced no audio samples")
    return samples


def chunk_audio(
    samples: npt.NDArray[np.float32],
    sample_rate: int,
    chunk_seconds: float,
    min_chunk_seconds: float,
) -> list[npt.NDArray[np.float32]]:
    """Split samples into non-overlapping windows of ``chunk_seconds``.

    CLAP consumes 10 s windows, so tracks are pre-chunked here instead of
    relying on the processor's silent truncation.

    Args:
        samples: Mono PCM samples from :func:`decode_audio`.
        sample_rate: Sample rate of ``samples`` in Hz.
        chunk_seconds: Window length in seconds.
        min_chunk_seconds: Trailing window shorter than this is dropped.

    Returns:
        List of chunk views. A track no longer than ``chunk_seconds`` yields
        exactly one chunk (the whole track); empty input yields no chunks.
    """
    chunk_size = int(sample_rate * chunk_seconds)
    min_size = int(sample_rate * min_chunk_seconds)
    if samples.size == 0:
        return []
    if samples.size <= chunk_size:
        return [samples]
    chunks = [samples[start : start + chunk_size] for start in range(0, samples.size, chunk_size)]
    if chunks[-1].size < min_size:
        del chunks[-1]
    return chunks
