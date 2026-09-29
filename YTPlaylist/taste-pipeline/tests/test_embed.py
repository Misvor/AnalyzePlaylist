"""Tests for taste_pipeline.embed: pooling/normalization math with a mocked CLAP pair."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch

from taste_pipeline import embed
from taste_pipeline.config import Config
from taste_pipeline.embed import EmbeddingError, embed_track

SAMPLE_RATE: int = 48000
EMBEDDING_DIM: int = 512


@dataclass(frozen=True, slots=True)
class _FakeAudioOutput:
    """Duck-type of the 5.x BaseModelOutputWithPooling the audio tower returns."""

    pooler_output: torch.Tensor


class _FakeProcessor:
    """Stand-in for ClapProcessor; batches whatever chunks it is handed into (batch, 1) features."""

    def __call__(
        self,
        *,
        audio: npt.NDArray[np.float32] | list[npt.NDArray[np.float32]],
        sampling_rate: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        _ = (sampling_rate, return_tensors)
        batch = len(audio) if isinstance(audio, list) else 1
        return {"input_features": torch.zeros(batch, 1)}


class _FakeModel:
    """Stand-in for ClapModel; replays a fixed (512,) vector sequence, one batch per forward pass."""

    def __init__(self, vectors: list[npt.NDArray[np.float32]]) -> None:
        self._vectors: list[npt.NDArray[np.float32]] = vectors
        self.calls: int = 0
        self.chunks: int = 0

    def get_audio_features(self, **inputs: torch.Tensor) -> _FakeAudioOutput:
        batch = int(inputs["input_features"].shape[0])
        selected = self._vectors[self.chunks : self.chunks + batch]
        self.chunks += batch
        self.calls += 1
        pooled = torch.from_numpy(np.stack(selected, axis=0))  # pyright: ignore[reportUnknownMemberType]  # torch stub gap
        return _FakeAudioOutput(pooler_output=pooled)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    data_dir = tmp_path / "data"
    return Config(
        like_library_dir=tmp_path,
        data_dir=data_dir,
        cookie_file=tmp_path / "cookies.txt",
        download_archive=data_dir / "yt-dlp-archive.txt",
        device="cpu",  # keep the mocked math tests device-agnostic and fast
    )


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch, model: _FakeModel, samples: npt.NDArray[np.float32]
) -> None:
    """Swap the CLAP pair and the ffmpeg decode for fakes; chunking stays real."""

    def fake_get_model(device: str = "auto") -> tuple[_FakeModel, _FakeProcessor]:
        _ = device
        return (model, _FakeProcessor())

    def fake_decode(_path: Path, _rate: int) -> npt.NDArray[np.float32]:
        return samples

    monkeypatch.setattr(embed, "get_model", fake_get_model)
    monkeypatch.setattr(embed, "decode_audio", fake_decode)


def test_embed_track_returns_unit_float32_vector(
    monkeypatch: pytest.MonkeyPatch, config: Config, tmp_path: Path
) -> None:
    # Given a 5 s track (one chunk) and a model returning a fixed vector
    samples = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)
    model = _FakeModel([np.arange(1, EMBEDDING_DIM + 1, dtype=np.float32)])
    _patch_pipeline(monkeypatch, model, samples)

    # When embedding the track
    result = embed_track(tmp_path / "track.wav", config)

    # Then the output is a unit-norm (512,) float32 vector and the model ran once
    assert result.shape == (EMBEDDING_DIM,)
    assert result.dtype == np.float32
    assert float(np.linalg.norm(result)) == pytest.approx(1.0)
    assert model.calls == 1


def test_embed_track_mean_pools_across_chunks(
    monkeypatch: pytest.MonkeyPatch, config: Config, tmp_path: Path
) -> None:
    # Given a 25 s track -> 3 chunks (10 s + 10 s + 5 s kept tail) and known per-chunk vectors
    samples = np.zeros(25 * SAMPLE_RATE, dtype=np.float32)
    v1 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    v2[1] = 1.0
    v3 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    v3[0] = 3.0
    model = _FakeModel([v1, v2, v3])
    _patch_pipeline(monkeypatch, model, samples)

    # When embedding the track
    result = embed_track(tmp_path / "track.wav", config)

    # Then the result is the L2-normalized mean of the three chunk vectors
    # mean = [4/3, 1/3, 0, ...] -> normalized = [4/sqrt(17), 1/sqrt(17), 0, ...]
    # and all three chunks go through a single batched forward pass.
    expected = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    expected[0] = 4.0 / math.sqrt(17.0)
    expected[1] = 1.0 / math.sqrt(17.0)
    assert model.chunks == 3
    assert model.calls == 1
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_embed_track_splits_chunks_into_batches(
    monkeypatch: pytest.MonkeyPatch, config: Config, tmp_path: Path
) -> None:
    # Given an 85 s track -> 9 chunks and 9 standard-basis vectors
    samples = np.zeros(85 * SAMPLE_RATE, dtype=np.float32)
    vectors = []
    for i in range(9):
        vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        vector[i] = 1.0
        vectors.append(vector)
    model = _FakeModel(vectors)
    _patch_pipeline(monkeypatch, model, samples)

    # When embedding the track
    result = embed_track(tmp_path / "track.wav", config)

    # Then all 9 chunks are processed in two forward passes (8 + 1) ...
    assert model.chunks == 9
    assert model.calls == 2
    # ... and the normalized mean of the 9 basis vectors is 1/3 on the first 9 dims.
    expected = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    expected[:9] = 1.0 / 3.0
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_embed_track_nan_embedding_raises_embedding_error(
    monkeypatch: pytest.MonkeyPatch, config: Config, tmp_path: Path
) -> None:
    # Given a model whose embedding contains NaN
    samples = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)
    vector = np.ones(EMBEDDING_DIM, dtype=np.float32)
    vector[7] = np.nan
    model = _FakeModel([vector])
    _patch_pipeline(monkeypatch, model, samples)
    audio_path = tmp_path / "track.wav"

    # When embedding, Then a typed EmbeddingError naming the file is raised
    with pytest.raises(EmbeddingError) as exc_info:
        _ = embed_track(audio_path, config)
    assert exc_info.value.path == audio_path


def test_embed_track_wrong_shape_raises_embedding_error(
    monkeypatch: pytest.MonkeyPatch, config: Config, tmp_path: Path
) -> None:
    # Given a model returning a 256-d embedding instead of 512-d
    samples = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)
    model = _FakeModel([np.ones(256, dtype=np.float32)])
    _patch_pipeline(monkeypatch, model, samples)
    audio_path = tmp_path / "track.wav"

    # When embedding, Then a typed EmbeddingError naming the file is raised
    with pytest.raises(EmbeddingError) as exc_info:
        _ = embed_track(audio_path, config)
    assert exc_info.value.path == audio_path


def test_resolve_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given CUDA is reported available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    # When resolving "auto", Then it picks cuda
    assert embed._resolve_device("auto") == "cuda"


def test_resolve_device_auto_prefers_mps_over_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given CUDA is unavailable but MPS is available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    # When resolving "auto", Then it picks mps
    assert embed._resolve_device("auto") == "mps"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given neither CUDA nor MPS is available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    # When resolving "auto", Then it falls back to cpu
    assert embed._resolve_device("auto") == "cpu"


def test_resolve_device_explicit_cuda_falls_back_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a stale cuda config on a box without CUDA
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # When resolving "cuda", Then it falls back to cpu instead of crashing
    assert embed._resolve_device("cuda") == "cpu"
