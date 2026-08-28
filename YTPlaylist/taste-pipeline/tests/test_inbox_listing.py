"""Tests for the Inbox triage UI: HTML page rendering + JSON listing.

Covers ``GET /triage`` (HTML page with one row per inbox FLAC, action
buttons wired to ``POST /api/triage/<video_id>``) and ``GET /api/triage``
(JSON listing used by htmx partials / scripted clients). The action
endpoint is in ``test_triage_actions.py``; the audio route is in
``test_audio_route.py``.

All tests use ``tmp_path`` for a fresh ``data_dir`` and never touch the
user's real data. The fake FLAC files are just a few bytes; the page
contract only checks rendering, not audio bytes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from conftest import _client, _make_config, _seed_inbox

if TYPE_CHECKING:
    from pathlib import Path


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
