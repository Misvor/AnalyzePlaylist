"""Download pass-C: FLAC audio download with metadata sidecars and download-archive.

Pass C runs only on candidates the detection cascade classified as songs. Each
candidate is downloaded through the yt-dlp Python API into ``data_dir/inbox``
as FLAC (``bestaudio`` source + ``FFmpegExtractAudio``), one video at a time so
a single failure cannot abort the batch. Unlike passes A and B (read-only),
this pass writes yt-dlp's ``download_archive``: it is the cross-week
re-download guard at the extractor level. After each successful download a
``<audio_path>.meta.json`` sidecar carrying the pass-B metadata is written next
to the FLAC and the state store records stage ``downloaded`` with the path.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from taste_pipeline.config import Config
from taste_pipeline.metadata import TrackMetadata, metadata_to_json
from taste_pipeline.state import StateStore

_LOGGER: Final = logging.getLogger(__name__)
_WATCH_URL: Final[str] = "https://www.youtube.com/watch?v={video_id}"
_OUTTMPL_NAME: Final[str] = "%(uploader).80s - %(title).80s [%(id)s].%(ext)s"
_FLAC_SUFFIX: Final[str] = ".flac"
_SIDECAR_SUFFIX: Final[str] = ".meta.json"


@dataclass(frozen=True, slots=True)
class DownloadedTrack:
    """One successfully downloaded candidate: the FLAC on disk plus its pass-B metadata."""

    audio_path: Path
    meta: TrackMetadata


def _ydl_options(config: Config) -> dict[str, object]:
    """Build the yt-dlp parameters for a FLAC download into the inbox with a download archive."""
    return {
        "format": "bestaudio/best",
        "cookiefile": str(config.cookie_file),
        "proxy": config.proxy,
        "outtmpl": str(config.data_dir / "inbox" / _OUTTMPL_NAME),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "flac"}],
        "download_archive": str(config.download_archive),
        "quiet": True,
        "noprogress": True,
    }


def _find_downloaded_file(inbox: Path, video_id: str) -> Path | None:
    """Locate the FLAC produced for ``video_id`` via the ``[id]`` marker our outtmpl embeds."""
    marker = f"[{video_id}]"
    matches = sorted(path for path in inbox.glob(f"*{_FLAC_SUFFIX}") if marker in path.stem)
    return matches[0] if matches else None


def _write_sidecar(audio_path: Path, meta: TrackMetadata) -> Path:
    """Write the pass-B metadata as ``<audio_path>.meta.json`` next to the FLAC."""
    sidecar = audio_path.with_name(audio_path.name + _SIDECAR_SUFFIX)
    payload = json.dumps(metadata_to_json(meta), indent=2, ensure_ascii=False)
    _ = sidecar.write_text(payload + "\n", encoding="utf-8")
    return sidecar


def download_songs(
    candidates: list[TrackMetadata], config: Config, state: StateStore
) -> list[DownloadedTrack]:
    """Download each candidate's audio as FLAC into the inbox, writing sidecars and state.

    Each video is downloaded individually so a per-video ``DownloadError``
    (private/deleted mid-week, region lock) is logged and skipped without
    aborting the batch. yt-dlp writes ``config.download_archive`` in this pass,
    so re-runs skip already-downloaded videos at the extractor level; the state
    store is the path-carrying record the later passes read.

    Args:
        candidates: Song candidates with pass-B metadata, in pipeline order.
        config: Pipeline configuration (cookie file, inbox, download archive).
        state: State store; each successful download is recorded with stage
            ``downloaded`` and the audio path.

    Returns:
        The successfully downloaded tracks, in input order.
    """
    inbox = config.data_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    results: list[DownloadedTrack] = []
    # See feed.py: typeshed's yt-dlp stub mistypes params we rely on; the
    # options dict carries the runtime values yt-dlp documents.
    with YoutubeDL(_ydl_options(config)) as ydl:  # pyright: ignore[reportArgumentType]
        for meta in candidates:
            try:
                _ = ydl.download([_WATCH_URL.format(video_id=meta.video_id)])
            except DownloadError as exc:
                _LOGGER.warning("skipping %s: download failed: %s", meta.video_id, exc)
                continue
            audio_path = _find_downloaded_file(inbox, meta.video_id)
            if audio_path is None:
                _LOGGER.warning("skipping %s: download succeeded but no FLAC found in inbox", meta.video_id)
                continue
            _ = _write_sidecar(audio_path, meta)
            _ = state.record_seen(meta.video_id, stage="downloaded")
            _ = state.record_download(meta.video_id, audio_path)
            results.append(DownloadedTrack(audio_path=audio_path, meta=meta))
    return results
