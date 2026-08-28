"""Tests for the Inbox triage action endpoint: ``POST /api/triage/{video_id}``.

Each action (``keep``/``review``/``skip``/``dislike``) moves the inbox FLAC
+ its sibling ``<file>.meta.json`` sidecar into the matching
``<data_dir>/<action>/`` subdir and REPLACES the ``downloads`` row's
``stage`` + ``audio_path`` via :meth:`StateStore.record_download`. The
endpoint returns 200 + ``{"ok": true, "new_stage": "..."}`` on success;
400 on an unknown action; 404 on an unknown ``video_id``; and is
idempotent on re-post.

The page rendering lives in ``test_inbox_listing.py``; the audio route
lives in ``test_audio_route.py``.
"""

from __future__ import annotations

from pathlib import Path

from conftest import _client, _make_config, _seed_inbox
from taste_pipeline.state import StateStore


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
