"""Slow tests for taste_pipeline.embed: real CLAP model on generated sine waves."""

import math
import shutil
import wave
from pathlib import Path
from typing import SupportsFloat, cast

import numpy as np
import numpy.typing as npt
import pytest

from taste_pipeline.config import Config
from taste_pipeline.embed import embed_track, get_model

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None,
        reason="ffmpeg is not on PATH; embedding tests decode via ffmpeg",
    ),
]

SAMPLE_RATE: int = 48000
EMBEDDING_DIM: int = 512


def _write_sine_wav(path: Path, *, frequency: float, seconds: float) -> None:
    """Write a mono 16-bit PCM sine WAV at 48 kHz."""
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    pcm = (0.5 * np.sin(2.0 * math.pi * frequency * t) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


@pytest.fixture(scope="module")
def clap_model() -> object:
    """Load the real CLAP model once per module; skip cleanly when the bundled weights are missing."""
    try:
        return get_model()
    except Exception as exc:  # noqa: BLE001  # any load failure (missing weights, file error) means "model unavailable"
        pytest.skip(f"bundled CLAP weights unavailable: {exc}")


@pytest.fixture
def config(tmp_path: Path) -> Config:
    data_dir = tmp_path / "data"
    return Config(
        like_library_dir=tmp_path,
        data_dir=data_dir,
        cookie_file=tmp_path / "cookies.txt",
        download_archive=data_dir / "yt-dlp-archive.txt",
    )


def test_real_model_embeds_10s_sine(clap_model: object, config: Config, tmp_path: Path) -> None:
    _ = clap_model  # requesting the fixture forces the real model to load (or skip)
    # Given a 10 s 440 Hz sine WAV at 48 kHz
    wav_path = tmp_path / "sine_440.wav"
    _write_sine_wav(wav_path, frequency=440.0, seconds=10.0)

    # When embedding with the real CLAP model
    vector = embed_track(wav_path, config)

    # Then the embedding is finite, (512,), float32, and approximately unit-norm
    assert vector.shape == (EMBEDDING_DIM,)
    assert vector.dtype == np.float32
    assert np.all(np.isfinite(vector))
    assert 0.9 <= float(np.linalg.norm(vector)) <= 1.1


def test_real_model_distinguishes_frequencies(clap_model: object, config: Config, tmp_path: Path) -> None:
    _ = clap_model  # requesting the fixture forces the real model to load (or skip)
    # Given two 10 s sines at clearly different pitches
    low_path = tmp_path / "sine_220.wav"
    high_path = tmp_path / "sine_880.wav"
    _write_sine_wav(low_path, frequency=220.0, seconds=10.0)
    _write_sine_wav(high_path, frequency=880.0, seconds=10.0)

    # When embedding both
    v_low = embed_track(low_path, config)
    v_high = embed_track(high_path, config)

    # Then their cosine similarity is well below 1 (the embeddings carry pitch information)
    assert _cosine_similarity(v_low, v_high) < 0.999


def _cosine_similarity(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> float:
    similarity = cast("SupportsFloat", np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(similarity)
