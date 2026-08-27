"""Tests for the Inbox triage UI: HTML page, action API, and audio streaming.

TDD-first for todo-9 (todo-9 in the orchestrator's task is "Inbox triage UI" — a
combined implementation that subsumes the plan's todos 9 (Inbox listing +
audio serving) and 10 (Triage action API) and the page-rendering half of
todo-11 (Triage page). All endpoints live under one new FastAPI router in
``taste_pipeline.web.routes_triage``.

The contract under test:

- ``GET /triage`` renders an HTML page that lists every file currently in
  ``<data_dir>/inbox/*.flac``, each row showing the filename, the title
  + uploader parsed from the sibling ``<file>.meta.json`` sidecar, an
  inline ``<audio controls preload="none" src="/audio/inbox/<file>">``
  player, and four action buttons (Keep / Review / Skip / Dislike). The
  buttons are wired with htmx so clicking posts to
  ``POST /api/triage/<video_id>`` and removes the row on success.

- ``POST /api/triage/<video_id>`` body ``{"action": "keep"|"review"|"skip"|"dislike"}``
  moves the inbox file (and its sidecar) to ``<data_dir>/<action>/`` via
  ``Path.rename`` and records the new stage via
  ``StateStore.record_download`` (REPLACE semantics). Returns 200 +
  ``{"ok": true, "new_stage": "..."}`` or 400 (unknown action) / 404
  (unknown video_id).

- ``GET /audio/<rel_path:path>`` streams the bytes of any file under
  ``<data_dir>/<rel_path>`` via ``FileResponse``. The rel_path is
  rejected when it contains a ``..`` segment (path-traversal guard).

All tests use ``tmp_path`` for a fresh ``data_dir`` and never touch the
user's real data. The fake FLAC files are just a few bytes — the audio
route streams bytes regardless of format, and the test only checks the
byte-for-byte echo.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from taste_pipeline.config import Config, load_config
from taste_pipeline.state import StateStore
from taste_pipeline.web import create_app


def _toml_path(path: Path) -> str:
    """Render a Path as a TOML-safe string (forward slashes avoid backslash escapes on Windows)."""
    return path.as_posix()


def _write_config(tmp_path: Path) -> Path:
    """Write a minimal valid config TOML under tmp_path and return its path.

    ``exist_ok=True`` is necessary because tests that already constructed a
    Config (e.g. to seed the data_dir) call this helper a second time to
    build a second app -- both app builds share the same ``like_lib/`` dir
    under tmp_path. Config validation also expects ``like_library_dir`` to
    exist; we never read its contents, the empty dir is enough.
    """
    like_lib = tmp_path / "lib"
    like_lib.mkdir(exist_ok=True)
    body = (
        f'like_library_dir = "{_toml_path(like_lib)}"\n'
        f'data_dir = "{_toml_path(tmp_path / "data")}"\n'
        f'cookie_file = "{_toml_path(tmp_path / "cookies.txt")}"\n'
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def _make_config(tmp_path: Path) -> Config:
    """Build a real Config via load_config from a tmp TOML file."""
    return load_config(_write_config(tmp_path))


def _seed_inbox(
    data_dir: Path,
    tracks: list[tuple[str, str, str]],
) -> list[Path]:
    """Write fake FLACs + sidecars for the given tracks and record them in the state store.

    Args:
        data_dir: Pipeline data dir (already has ``inbox/`` from config load).
        tracks: Each entry is ``(video_id, title, uploader)``.

    Returns:
        The list of absolute paths to the written FLAC files (in input order).
    """
    inbox = data_dir / "inbox"
    store = StateStore(data_dir)
    try:
        paths: list[Path] = []
        for video_id, title, uploader in tracks:
            # Filename embeds the [video_id] marker the pipeline uses for lookup.
            safe_title = re.sub(r"[^A-Za-z0-9_-]", "_", title)[:40]
            audio_path = inbox / f"{safe_title} [{video_id}].flac"
            _ = audio_path.write_bytes(b"FAKE_FLAC_HEADER" + b"\x00" * 32)
            sidecar = audio_path.with_name(audio_path.name + ".meta.json")
            sidecar_payload = {
                "video_id": video_id,
                "title": title,
                "uploader": uploader,
                "uploader_id": None,
                "channel_id": None,
                "upload_date": None,
                "timestamp": None,
                "categories": None,
                "duration": None,
                "description": None,
                "track": title,
                "artist": uploader,
                "album": None,
                "thumbnail_url": None,
                "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
            }
            _ = sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
            _ = store.record_download(video_id, audio_path)
            paths.append(audio_path)
    finally:
        store.close()
    return paths


def _client(tmp_path: Path) -> TestClient:
    """Build a TestClient over a real Config-derived app."""
    return TestClient(create_app(_make_config(tmp_path)))


# ── HTML page contract ────────────────────────────────────────────────────


def test_get_triage_returns_inbox_files_listed_with_action_buttons(tmp_path: Path) -> None:
    # Given a tmp data_dir with two seeded inbox tracks
    cfg = _make_config(tmp_path)
    _ = _seed_inbox(
        cfg.data_dir,
        [("aaaaaaaaaaa1", "Song One", "Artist One"), ("aaaaaaaaaaa2", "Song Two", "Artist Two")],
    )
    client = _client(tmp_path)

    # When requesting the triage page
    response = client.get("/triage")

    # Then the response is 200 HTML and lists both files with 4 action buttons each
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "aaaaaaaaaaa1" in body, "first video_id missing from triage HTML"
    assert "aaaaaaaaaaa2" in body, "second video_id missing from triage HTML"
    assert ".flac" in body, "triage HTML does not mention .flac files"

    # Action buttons per row: count hx-post="/api/triage/" occurrences. 2 rows * 4 buttons = 8.
    button_post_count = body.count('hx-post="/api/triage/')
    assert button_post_count == 8, (
        f"expected 8 hx-post /api/triage/ buttons (2 rows x 4 actions), got {button_post_count}"
    )
    # Each of the 4 action verbs must appear in some hx-vals JSON payload
    for verb in ("keep", "review", "skip", "dislike"):
        assert f'"action": "{verb}"' in body, (
            f"triage HTML has no action={verb!r} hx-vals; expected one button per row per verb"
        )


def test_get_triage_shows_metadata_from_sidecar(tmp_path: Path) -> None:
    # Given a tmp data_dir with one seeded inbox track carrying unique title/uploader strings
    cfg = _make_config(tmp_path)
    _ = _seed_inbox(
        cfg.data_dir,
        [("zzzzzzzzzzz9", "My Unique Song Title XYZ", "My Unique Artist ABC")],
    )
    client = _client(tmp_path)

    # When requesting the triage page
    response = client.get("/triage")

    # Then the rendered HTML contains the title and uploader parsed from the .meta.json sidecar
    assert response.status_code == 200
    body = response.text
    assert "My Unique Song Title XYZ" in body, (
        "triage HTML missing the title parsed from the .meta.json sidecar"
    )
    assert "My Unique Artist ABC" in body, (
        "triage HTML missing the uploader parsed from the .meta.json sidecar"
    )


def test_get_triage_uses_base_template(tmp_path: Path) -> None:
    # Given an app built from a valid config with one seeded inbox track
    cfg = _make_config(tmp_path)
    _ = _seed_inbox(cfg.data_dir, [("bbbbbbbbbbb1", "Anything", "Whoever")])
    client = _client(tmp_path)

    # When requesting the triage page
    response = client.get("/triage")

    # Then the page inherits base.html's structural IDs (action-bar, main-nav, content)
    # and the nav anchors (dashboard, triage, index, settings) are present.
    assert response.status_code == 200
    body = response.text
    for required_id in ("action-bar", "main-nav", "content", "status-bar"):
        assert f'id="{required_id}"' in body, (
            f"triage page missing inherited base.html element id={required_id!r}"
        )
    for nav_id in ("nav-dashboard", "nav-triage", "nav-index", "nav-settings"):
        assert nav_id in body, f"triage page missing nav anchor {nav_id!r}"


def test_get_triage_audio_has_preload_none_to_avoid_preloading_all(tmp_path: Path) -> None:
    # Given an app with two seeded inbox tracks
    cfg = _make_config(tmp_path)
    _ = _seed_inbox(
        cfg.data_dir,
        [("ccccccccccc1", "Song A", "Artist A"), ("ccccccccccc2", "Song B", "Artist B")],
    )
    client = _client(tmp_path)

    # When requesting the triage page
    response = client.get("/triage")

    # Then every <audio> element carries preload="none" or preload="metadata" (so the browser
    # does not pre-buffer every FLAC on page load — that would defeat the point of the triage
    # page for an inbox with 100s of tracks). One <audio> per row, so 2 total.
    assert response.status_code == 200
    body = response.text
    audio_tags = re.findall(r"<audio\b[^>]*>", body)
    assert len(audio_tags) == 2, f"expected 2 <audio> elements, found {len(audio_tags)}: {audio_tags}"
    for tag in audio_tags:
        assert 'preload="none"' in tag or 'preload="metadata"' in tag, (
            f"<audio> tag missing preload hint: {tag!r}"
        )
        # And the src points at /audio/inbox/...
        assert 'src="/audio/inbox/' in tag, f"<audio> src not under /audio/inbox/: {tag!r}"


def test_get_triage_button_has_hx_target_closest_tr_and_swap_outerhtml(tmp_path: Path) -> None:
    # Given an app with one seeded inbox track
    cfg = _make_config(tmp_path)
    _ = _seed_inbox(cfg.data_dir, [("ddddddddddd1", "Demo", "Demoer")])
    client = _client(tmp_path)

    # When requesting the triage page
    response = client.get("/triage")

    # Then every action button has hx-target="closest tr" + hx-swap="outerHTML" so clicking
    # removes the entire row from the DOM (one action per track = one row gone).
    assert response.status_code == 200
    body = response.text
    # At least one button per action verb carries the row-removal pattern.
    for verb in ("keep", "review", "skip", "dislike"):
        pattern = (
            r'<button\b[^>]*\bhx-post="/api/triage/[^"]+"[^>]*\bhx-vals=\'{"action": "'
            + verb
            + r'"}\'[^>]*\bhx-target="closest tr"[^>]*\bhx-swap="outerHTML"'
        )
        assert re.search(pattern, body), (
            f"button for action={verb!r} missing hx-target='closest tr' + hx-swap='outerHTML'"
        )


# ── Action endpoint contract ───────────────────────────────────────────────


def test_post_triage_keep_moves_file_to_keep_subdir_and_updates_stage(tmp_path: Path) -> None:
    # Given a tmp data_dir with one seeded inbox track (stage=downloaded)
    cfg = _make_config(tmp_path)
    seeded = _seed_inbox(cfg.data_dir, [("eeeeeeeeeee1", "Keep Me", "Artist")])
    inbox_path = seeded[0]
    keep_dir = cfg.data_dir / "keep"
    client = _client(tmp_path)

    # Sanity: file starts in inbox/, not in keep/
    assert inbox_path.exists()
    assert not (keep_dir / inbox_path.name).exists()

    # When POSTing keep for this video_id
    response = client.post("/api/triage/eeeeeeeeeee1", json={"action": "keep"})

    # Then 200 + {"ok": true, ...}, file moved to keep/, sidecar moved too, state.stage="kept"
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("ok") is True
    assert payload.get("new_stage") == "kept"

    new_path = keep_dir / inbox_path.name
    assert new_path.exists(), f"FLAC not moved to keep/: {new_path}"
    assert not inbox_path.exists(), f"FLAC still in inbox/: {inbox_path}"
    assert new_path.with_name(new_path.name + ".meta.json").exists(), "sidecar not moved with FLAC"

    store = StateStore(cfg.data_dir)
    try:
        with store._conn:
            row = store._conn.execute(
                "SELECT stage, audio_path FROM downloads WHERE video_id = ?", ("eeeeeeeeeee1",)
            ).fetchone()
    finally:
        store.close()
    assert row is not None, "downloads row missing after triage"
    stage_value, audio_path_value = row
    assert stage_value == "kept", f"downloads.stage is {stage_value!r}, expected 'kept'"
    assert Path(audio_path_value).name == new_path.name, (
        f"downloads.audio_path = {audio_path_value!r}, expected to point at {new_path!r}"
    )


def test_post_triage_review_skip_dislike_similar(tmp_path: Path) -> None:
    # Each action moves the FLAC to the right subdir AND sets the right stage.
    cases = [
        ("fffffffffff1", "review", "review", "review/"),
        ("fffffffffff2", "skip", "skipped", "skip/"),
        ("fffffffffff3", "dislike", "disliked", "dislike/"),
    ]
    for video_id, action, expected_stage, expected_subdir in cases:
        # Given a tmp data_dir (fresh per iteration via the tmp_path fixture)
        sub_tmp = tmp_path / video_id
        sub_tmp.mkdir()
        cfg = _make_config(sub_tmp)
        seeded = _seed_inbox(cfg.data_dir, [(video_id, f"Song {action}", f"Artist {action}")])
        inbox_path = seeded[0]
        target_dir = cfg.data_dir / expected_subdir.rstrip("/")
        client = _client(sub_tmp)

        # When POSTing the action
        response = client.post(f"/api/triage/{video_id}", json={"action": action})

        # Then 200 + file in <action>/ + state.stage = expected_stage
        assert response.status_code == 200, (
            f"action={action!r} returned {response.status_code}, body={response.text}"
        )
        assert response.json().get("new_stage") == expected_stage
        new_path = target_dir / inbox_path.name
        assert new_path.exists(), f"FLAC not moved to {expected_subdir}: {new_path}"
        assert not inbox_path.exists(), f"FLAC still in inbox/: {inbox_path}"

        store = StateStore(cfg.data_dir)
        try:
            with store._conn:
                row = store._conn.execute(
                    "SELECT stage FROM downloads WHERE video_id = ?", (video_id,)
                ).fetchone()
        finally:
            store.close()
        assert row is not None
        assert row[0] == expected_stage, f"action={action!r}: stage={row[0]!r}, expected {expected_stage!r}"


def test_post_triage_unknown_action_returns_400(tmp_path: Path) -> None:
    # Given a tmp data_dir with one seeded track
    cfg = _make_config(tmp_path)
    _ = _seed_inbox(cfg.data_dir, [("ggggggggggg1", "Whatever", "Whoever")])
    client = _client(tmp_path)

    # When POSTing an unknown action
    response = client.post("/api/triage/ggggggggggg1", json={"action": "bogus"})

    # Then 400 (the body parser accepts str but the handler rejects unknown verbs)
    assert response.status_code == 400, (
        f"unknown action returned {response.status_code}, expected 400; body={response.text}"
    )


def test_post_triage_unknown_video_id_returns_404(tmp_path: Path) -> None:
    # Given a tmp data_dir with NO matching video_id
    cfg = _make_config(tmp_path)
    _ = _seed_inbox(cfg.data_dir, [("hhhhhhhhhhh1", "X", "Y")])
    client = _client(tmp_path)

    # When POSTing keep for an unknown video_id
    response = client.post("/api/triage/nonexistent9999", json={"action": "keep"})

    # Then 404
    assert response.status_code == 404, (
        f"unknown video_id returned {response.status_code}, expected 404; body={response.text}"
    )


def test_post_triage_idempotent_repeat_returns_200(tmp_path: Path) -> None:
    # Given a tmp data_dir with one seeded track
    cfg = _make_config(tmp_path)
    seeded = _seed_inbox(cfg.data_dir, [("iiiiiiiiiii1", "Idem", "Artist")])
    client = _client(tmp_path)

    # When POSTing keep twice
    first = client.post("/api/triage/iiiiiiiiiii1", json={"action": "keep"})
    second = client.post("/api/triage/iiiiiiiiiii1", json={"action": "keep"})

    # Then both return 200 (the second is a no-op: file already in keep/, state stays "kept")
    assert first.status_code == 200
    assert second.status_code == 200, (
        f"second POST returned {second.status_code}, expected 200 (idempotent); body={second.text}"
    )
    assert (cfg.data_dir / "keep" / seeded[0].name).exists()


# ── Audio route contract ───────────────────────────────────────────────────


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


# ── JSON inbox listing (optional /api/triage endpoint) ─────────────────────


def test_get_api_triage_returns_json_list_of_inbox_items(tmp_path: Path) -> None:
    # Given a tmp data_dir with two seeded tracks
    cfg = _make_config(tmp_path)
    _ = _seed_inbox(
        cfg.data_dir,
        [("kkkkkkkkkkk1", "Listed A", "Artist A"), ("kkkkkkkkkkk2", "Listed B", "Artist B")],
    )
    client = _client(tmp_path)

    # When requesting the JSON inbox listing
    response = client.get("/api/triage")

    # Then 200 + JSON array of length 2 with the expected fields
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == 2
    ids = {item["video_id"] for item in payload}
    assert ids == {"kkkkkkkkkkk1", "kkkkkkkkkkk2"}
    for item in payload:
        assert "filename" in item
        assert item["filename"].endswith(".flac")
        assert "title" in item
        assert "uploader" in item
        assert "audio_path" in item
