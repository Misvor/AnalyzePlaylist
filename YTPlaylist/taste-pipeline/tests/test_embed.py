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
    """Stand-in for ClapProcessor; the tensor contents are irrelevant to the math under test."""

    def __call__(
        self, *, audio: npt.NDArray[np.float32], sampling_rate: int, return_tensors: str
    ) -> dict[str, torch.Tensor]:
        _ = (audio, sampling_rate, return_tensors)
        return {"input_features": torch.zeros(1, 1)}


class _FakeModel:
    """Stand-in for ClapModel; replays a fixed sequence of (512,) vectors, one per call."""

    def __init__(self, vectors: list[npt.NDArray[np.float32]]) -> None:
        self._vectors: list[npt.NDArray[np.float32]] = vectors
        self.calls: int = 0

    def get_audio_features(self, **_inputs: torch.Tensor) -> _FakeAudioOutput:
        vector = self._vectors[self.calls % len(self._vectors)]
        self.calls += 1
        pooled = torch.from_numpy(vector).unsqueeze(0)  # pyright: ignore[reportUnknownMemberType]  # torch stub gap
        return _FakeAudioOutput(pooler_output=pooled)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    data_dir = tmp_path / "data"
    return Config(
        like_library_dir=tmp_path,
        data_dir=data_dir,
        cookie_file=tmp_path / "cookies.txt",
        download_archive=data_dir / "yt-dlp-archive.txt",
    )


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch, model: _FakeModel, samples: npt.NDArray[np.float32]
) -> None:
    """Swap the CLAP pair and the ffmpeg decode for fakes; chunking stays real."""

    def fake_get_model(_name: str) -> tuple[_FakeModel, _FakeProcessor]:
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
    expected = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    expected[0] = 4.0 / math.sqrt(17.0)
    expected[1] = 1.0 / math.sqrt(17.0)
    assert model.calls == 3
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
