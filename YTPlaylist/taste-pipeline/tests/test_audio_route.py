"""Tests for the Inbox triage audio route: ``GET /audio/{rel_path:path}``.

The route streams bytes of any file under ``<data_dir>/<rel_path>`` via
``FileResponse``. The path-traversal guard rejects any ``..`` segment in
the raw input AND any resolved path that escapes ``data_dir`` (two
independent checks -- the resolved-prefix check catches symlink-driven
escapes the raw-segment check misses).

The page rendering lives in ``test_inbox_listing.py``; the triage action
endpoint lives in ``test_triage_actions.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from conftest import _client, _make_config

if TYPE_CHECKING:
    from pathlib import Path


def test_audio_route_streams_files_from_data_subdirs(tmp_path: Path) -> None:
    # Given a tmp data_dir with a fake FLAC in inbox/
    cfg = _make_config(tmp_path)
    fake_bytes = b"FAKE_FLAC_HEADER" + b"\x00" * 32
    inbox = cfg.data_dir / "inbox"
    audio_file = inbox / "test_song [jjjjjjjjjjj1].flac"
    _ = audio_file.write_bytes(fake_bytes)
    client = _client(tmp_path)

    # When requesting /audio/inbox/<file>
    response = client.get(f"/audio/inbox/{audio_file.name}")

    # Then 200 with a FLAC audio MIME type and the file bytes echoed back
    assert response.status_code == 200
    ct = response.headers["content-type"]
    assert ct.startswith(("audio/flac", "audio/x-flac", "application/octet-stream")), ct
    assert response.content == fake_bytes, "audio route did not echo file bytes verbatim"


def test_audio_route_rejects_path_traversal_with_dotdot(tmp_path: Path) -> None:
    # Given a tmp data_dir with the standard subdirs (inbox, keep, etc.)
    _ = _make_config(tmp_path)
    client = _client(tmp_path)

    # When requesting a path that escapes data_dir via ".."
    # /audio/inbox/../config.toml -> resolved should land OUTSIDE data_dir.
    response = client.get("/audio/inbox/../config.toml")

    # Then the request is rejected (400 or 404 — anything that is NOT a 200 with file bytes)
    assert response.status_code in (400, 403, 404), (
        f"path-traversal attempt returned {response.status_code}, expected rejection; body={response.text}"
    )
    # Specifically: the route MUST NOT have served config.toml. Status 200 with toml body
    # would be a real bypass; everything else is a correctly-defended endpoint.
    assert response.status_code != 200


def test_audio_route_unknown_file_returns_404(tmp_path: Path) -> None:
    # Given a tmp data_dir with NO matching file
    _ = _make_config(tmp_path)
    client = _client(tmp_path)

    # When requesting a path under inbox/ that doesn't exist on disk
    response = client.get("/audio/inbox/does_not_exist.flac")

    # Then 404
    assert response.status_code == 404
