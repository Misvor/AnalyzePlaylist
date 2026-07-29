"""Tests for taste_pipeline.audio: ffmpeg-based decode and fixed-window chunking."""

import math
import shutil
import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from taste_pipeline.audio import DecodeError, chunk_audio, decode_audio

SAMPLE_RATE: int = 48000
CHUNK_SECONDS: float = 10.0
MIN_CHUNK_SECONDS: float = 3.0
CHUNK_SIZE: int = SAMPLE_RATE * int(CHUNK_SECONDS)


def _write_sine_wav(path: Path, seconds: float, sample_rate: int = SAMPLE_RATE) -> None:
    t = np.arange(int(seconds * sample_rate), dtype=np.float64) / sample_rate
    pcm = (0.5 * np.sin(2.0 * math.pi * 440.0 * t) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def _chunk(
    signals: npt.NDArray[np.float32], min_chunk_seconds: float = MIN_CHUNK_SECONDS
) -> list[npt.NDArray[np.float32]]:
    return chunk_audio(signals, SAMPLE_RATE, CHUNK_SECONDS, min_chunk_seconds)


_REQUIRES_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg is not on PATH; decode tests require it (e.g. `scoop install ffmpeg`)",
)


def test_ffmpeg_is_on_path() -> None:
    # Given the environment PATH
    # When resolving the ffmpeg executable
    ffmpeg = shutil.which("ffmpeg")

    # Then it resolves to an invocable binary; absent ffmpeg skips rather than fails
    if ffmpeg is None:
        pytest.skip("ffmpeg is not on PATH; install it to enable decoding (e.g. `scoop install ffmpeg`)")
    assert Path(ffmpeg).name.lower().startswith("ffmpeg")


@_REQUIRES_FFMPEG
def test_decode_sine_returns_float32_mono(tmp_path: Path) -> None:
    # Given a 3 s mono sine WAV at 48 kHz
    wav_path = tmp_path / "sine_3s.wav"
    _write_sine_wav(wav_path, seconds=3.0)

    # When decoding at 48 kHz
    samples = decode_audio(wav_path, SAMPLE_RATE)

    # Then the PCM is 1-D (mono), float32, full length, and carries the sine
    assert samples.ndim == 1
    assert samples.shape == (3 * SAMPLE_RATE,)
    assert samples.dtype == np.float32
    assert float(np.max(samples)) == pytest.approx(0.5, abs=0.01)


@_REQUIRES_FFMPEG
def test_decode_then_chunk_25s_keeps_tail_chunk(tmp_path: Path) -> None:
    # Given a 25 s sine WAV decoded to PCM
    wav_path = tmp_path / "sine_25s.wav"
    _write_sine_wav(wav_path, seconds=25.0)
    samples = decode_audio(wav_path, SAMPLE_RATE)

    # When chunking with the defaults (10 s chunks, 3 s minimum tail)
    chunks = _chunk(samples)

    # Then the 5 s tail is kept (5 s >= 3 s minimum), yielding 3 chunks.
    # NOTE: the plan text mentions "25 s -> 2 chunks", but with the plan's own
    # defaults (chunk=10 s, min_chunk=3 s) the 5 s tail survives the drop rule.
    assert samples.shape == (25 * SAMPLE_RATE,)
    assert [chunk.shape for chunk in chunks] == [(CHUNK_SIZE,), (CHUNK_SIZE,), (5 * SAMPLE_RATE,)]
    assert all(chunk.dtype == np.float32 and chunk.ndim == 1 for chunk in chunks)


@_REQUIRES_FFMPEG
def test_decode_then_chunk_short_track_yields_single_chunk(tmp_path: Path) -> None:
    # Given a 0.5 s sine WAV decoded to PCM
    wav_path = tmp_path / "sine_0_5s.wav"
    _write_sine_wav(wav_path, seconds=0.5)
    samples = decode_audio(wav_path, SAMPLE_RATE)

    # When chunking a track shorter than one chunk
    chunks = _chunk(samples)

    # Then exactly one chunk covering the whole track is returned
    assert len(chunks) == 1
    assert chunks[0].shape == (int(0.5 * SAMPLE_RATE),)


@_REQUIRES_FFMPEG
def test_decode_garbage_file_raises_decode_error(tmp_path: Path) -> None:
    # Given a file whose bytes are not any audio format
    garbage_path = tmp_path / "garbage.bin"
    _ = garbage_path.write_bytes(b"\x00\x11\x22\x33 not an audio stream \xde\xad\xbe\xef" * 64)

    # When decoding, Then a typed DecodeError naming the file is raised
    with pytest.raises(DecodeError) as exc_info:
        _ = decode_audio(garbage_path, SAMPLE_RATE)
    assert exc_info.value.path == garbage_path


def test_chunk_audio_drops_tail_below_minimum() -> None:
    # Given 25 s of PCM and a 6 s minimum (tail 5 s < 6 s must be dropped)
    samples = np.zeros(25 * SAMPLE_RATE, dtype=np.float32)

    # When chunking
    chunks = _chunk(samples, min_chunk_seconds=6.0)

    # Then only the two full 10 s windows survive (the plan's QA shape)
    assert [chunk.shape for chunk in chunks] == [(CHUNK_SIZE,), (CHUNK_SIZE,)]


def test_chunk_audio_exact_multiple_produces_no_tail() -> None:
    # Given 20 s of PCM — an exact multiple of the 10 s window
    samples = np.zeros(20 * SAMPLE_RATE, dtype=np.float32)

    # When chunking
    chunks = _chunk(samples)

    # Then there is no tail to drop
    assert [chunk.shape for chunk in chunks] == [(CHUNK_SIZE,), (CHUNK_SIZE,)]


def test_chunk_audio_tail_at_minimum_is_kept() -> None:
    # Given 23 s of PCM — the 3 s tail is exactly the minimum
    samples = np.zeros(23 * SAMPLE_RATE, dtype=np.float32)

    # When chunking
    chunks = _chunk(samples)

    # Then the boundary tail is kept (drop rule is strictly "shorter than")
    assert [chunk.shape for chunk in chunks] == [(CHUNK_SIZE,), (CHUNK_SIZE,), (3 * SAMPLE_RATE,)]


def test_chunk_audio_short_track_returns_whole_track() -> None:
    # Given 0.5 s of PCM, shorter than one chunk
    samples = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)

    # When chunking
    chunks = _chunk(samples)

    # Then the whole track is the single chunk
    assert len(chunks) == 1
    assert chunks[0].shape == samples.shape


def test_chunk_audio_empty_input_returns_no_chunks() -> None:
    # Given an empty PCM array
    samples = np.zeros(0, dtype=np.float32)

    # When chunking, Then there is nothing to chunk
    assert _chunk(samples) == []
