"""Tests for taste_pipeline.config: TOML loading and eager validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from taste_pipeline.config import Config, ConfigError, load_config

if TYPE_CHECKING:
    from pathlib import Path


def _write_config(tmp_path: Path, body: str, name: str = "config.toml") -> Path:
    """Write a TOML body to ``tmp_path/name`` and return the path."""
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _toml_path(path: Path) -> str:
    """Render a Path as a TOML-safe string: forward slashes avoid backslash-escape pitfalls on Windows."""
    return path.as_posix()


def _minimal_body(like_library_dir: Path, data_dir: Path, cookie_file: Path) -> str:
    """A minimal valid config body with only the three required path fields."""
    return (
        f'like_library_dir = "{_toml_path(like_library_dir)}"\n'
        f'data_dir = "{_toml_path(data_dir)}"\n'
        f'cookie_file = "{_toml_path(cookie_file)}"\n'
    )


def test_minimal_config_loads_with_all_defaults(tmp_path: Path) -> None:
    # Given a config with only the three required path fields, all pointing inside tmp_path
    like_lib = tmp_path / "lib"
    like_lib.mkdir()
    data_dir = tmp_path / "data"
    cookie_file = tmp_path / "cookies.txt"
    config_path = _write_config(tmp_path, _minimal_body(like_lib, data_dir, cookie_file))

    # When loading it
    cfg = load_config(config_path)

    # Then every default value is applied, paths are absolute, and the
    # dataclass is the frozen+slooted one with all 16 fields.
    assert isinstance(cfg, Config)
    assert cfg.like_library_dir == like_lib.resolve()
    assert cfg.data_dir == data_dir.resolve()
    assert cfg.cookie_file == cookie_file.resolve()
    assert cfg.download_archive == (data_dir / "yt-dlp-archive.txt").resolve()
    assert cfg.model_name == "laion/larger_clap_music_and_speech"
    assert cfg.feed_window_days == 7
    assert cfg.max_feed_items == 500
    assert cfg.max_metadata_fetch == 150
    assert cfg.chunk_seconds == 10.0
    assert cfg.min_chunk_seconds == 3.0
    assert cfg.sample_rate == 48000
    assert cfg.keep_threshold is None
    assert cfg.skip_threshold is None
    assert cfg.min_dislikes_for_classifier == 30
    assert cfg.web_host == "127.0.0.1"
    assert cfg.web_port == 8741


def test_missing_like_library_dir_named_in_error(tmp_path: Path) -> None:
    # Given a config that omits `like_library_dir` (but has the other two required paths)
    data_dir = tmp_path / "data"
    cookie_file = tmp_path / "cookies.txt"
    config_path = _write_config(
        tmp_path,
        f'data_dir = "{_toml_path(data_dir)}"\ncookie_file = "{_toml_path(cookie_file)}"\n',
    )

    # When loading it
    # Then a ConfigError names the missing key
    with pytest.raises(ConfigError) as exc_info:
        load_config(config_path)
    assert "like_library_dir" in str(exc_info.value)
    assert "required" in str(exc_info.value)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    # Given a valid required trio plus an unknown extra key
    like_lib = tmp_path / "lib"
    like_lib.mkdir()
    config_path = _write_config(
        tmp_path,
        _minimal_body(like_lib, tmp_path / "data", tmp_path / "cookies.txt") + "bogus_key = 42\n",
    )

    # When loading
    # Then a ConfigError names the unknown key (and no other errors)
    with pytest.raises(ConfigError) as exc_info:
        load_config(config_path)
    assert "bogus_key" in str(exc_info.value)
    assert "unknown key" in str(exc_info.value)


def test_invalid_type_for_feed_window_days_rejected(tmp_path: Path) -> None:
    # Given a config where feed_window_days is a string instead of an int
    like_lib = tmp_path / "lib"
    like_lib.mkdir()
    config_path = _write_config(
        tmp_path,
        _minimal_body(like_lib, tmp_path / "data", tmp_path / "cookies.txt") + 'feed_window_days = "seven"\n',
    )

    # When loading (this is the plan's QA failure scenario)
    # Then a ConfigError names feed_window_days
    with pytest.raises(ConfigError) as exc_info:
        load_config(config_path)
    assert "feed_window_days" in str(exc_info.value)


def test_like_library_dir_must_exist(tmp_path: Path) -> None:
    # Given a config whose like_library_dir does not exist on disk
    nonexistent = tmp_path / "no_such_library"
    config_path = _write_config(
        tmp_path,
        _minimal_body(nonexistent, tmp_path / "data", tmp_path / "cookies.txt"),
    )

    # When loading
    # Then a ConfigError names like_library_dir and states it is missing
    with pytest.raises(ConfigError) as exc_info:
        load_config(config_path)
    message = str(exc_info.value)
    assert "like_library_dir" in message
    assert "does not exist" in message


def test_data_dir_created_with_subdirs(tmp_path: Path) -> None:
    # Given a config whose data_dir does not exist yet
    like_lib = tmp_path / "lib"
    like_lib.mkdir()
    data_dir = tmp_path / "new_data"
    assert not data_dir.exists()
    config_path = _write_config(
        tmp_path,
        _minimal_body(like_lib, data_dir, tmp_path / "cookies.txt"),
    )

    # When loading
    cfg = load_config(config_path)

    # Then the data dir AND every standard subdir now exist
    assert data_dir.is_dir()
    for sub in ("inbox", "keep", "review", "skip", "dislike", "index", "runs"):
        assert (data_dir / sub).is_dir(), f"missing subdir {sub}"
    assert cfg.data_dir == data_dir.resolve()


def test_data_dir_existing_subdirs_untouched(tmp_path: Path) -> None:
    # Given a data_dir that already exists and contains a sentinel file in a subdir
    like_lib = tmp_path / "lib"
    like_lib.mkdir()
    data_dir = tmp_path / "data"
    inbox = data_dir / "inbox"
    inbox.mkdir(parents=True)
    sentinel = inbox / "DO_NOT_DELETE.txt"
    sentinel.write_text("keep me", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        _minimal_body(like_lib, data_dir, tmp_path / "cookies.txt"),
    )

    # When loading
    load_config(config_path)

    # Then existing contents are preserved (mkdir(exist_ok=True) is non-destructive)
    assert sentinel.is_file()
    assert sentinel.read_text(encoding="utf-8") == "keep me"


def test_relative_paths_resolved_against_config_dir(tmp_path: Path) -> None:
    # Given a like_library_dir that exists inside tmp_path and a config
    # that references it with a relative `./` path
    like_lib = tmp_path / "lib"
    like_lib.mkdir()
    config_path = _write_config(
        tmp_path,
        'like_library_dir = "./lib"\ndata_dir = "./data"\ncookie_file = "./cookies.txt"\n',
    )

    # When loading
    cfg = load_config(config_path)

    # Then every path is absolute and points where we expect
    assert cfg.like_library_dir.is_absolute()
    assert cfg.data_dir.is_absolute()
    assert cfg.cookie_file.is_absolute()
    assert cfg.like_library_dir == like_lib.resolve()
    assert cfg.cookie_file == (tmp_path / "cookies.txt").resolve()


def test_download_archive_override(tmp_path: Path) -> None:
    # Given a config that sets download_archive explicitly
    like_lib = tmp_path / "lib"
    like_lib.mkdir()
    archive = tmp_path / "my-archive.txt"
    config_path = _write_config(
        tmp_path,
        _minimal_body(like_lib, tmp_path / "data", tmp_path / "cookies.txt")
        + f'download_archive = "{_toml_path(archive)}"\n',
    )

    # When loading
    cfg = load_config(config_path)

    # Then download_archive reflects the override (parent dir created via resolve)
    assert cfg.download_archive == archive.resolve()


def test_keep_skip_thresholds_optional_float(tmp_path: Path) -> None:
    # Given a config that sets both thresholds to explicit floats
    like_lib = tmp_path / "lib"
    like_lib.mkdir()
    config_path = _write_config(
        tmp_path,
        _minimal_body(like_lib, tmp_path / "data", tmp_path / "cookies.txt")
        + "keep_threshold = 0.65\nskip_threshold = 0.35\n",
    )

    # When loading
    cfg = load_config(config_path)

    # Then the values are stored as floats, not None
    assert cfg.keep_threshold == 0.65
    assert cfg.skip_threshold == 0.35


def test_multiple_errors_aggregated_in_one_message(tmp_path: Path) -> None:
    # Given a config with several problems at once
    config_path = _write_config(
        tmp_path,
        'feed_window_days = "seven"\nweb_port = 999999\nbogus_key = 1\n',
    )

    # When loading
    with pytest.raises(ConfigError) as exc_info:
        load_config(config_path)
    message = str(exc_info.value)

    # Then every invalid key is named in the single message
    assert "feed_window_days" in message
    assert "web_port" in message
    assert "bogus_key" in message
    # And the required-path-field errors also appear
    assert "like_library_dir" in message
    assert "data_dir" in message
    assert "cookie_file" in message


def test_min_chunk_seconds_must_not_exceed_chunk_seconds(tmp_path: Path) -> None:
    # Given chunk_seconds smaller than min_chunk_seconds
    like_lib = tmp_path / "lib"
    like_lib.mkdir()
    config_path = _write_config(
        tmp_path,
        _minimal_body(like_lib, tmp_path / "data", tmp_path / "cookies.txt")
        + "chunk_seconds = 2.0\nmin_chunk_seconds = 5.0\n",
    )

    # When loading
    # Then a ConfigError names the cross-field violation
    with pytest.raises(ConfigError) as exc_info:
        load_config(config_path)
    assert "min_chunk_seconds" in str(exc_info.value)
    assert "chunk_seconds" in str(exc_info.value)


def test_missing_config_file_raises_config_error(tmp_path: Path) -> None:
    # Given a path that does not exist
    # When loading
    # Then a ConfigError names the missing path
    with pytest.raises(ConfigError) as exc_info:
        load_config(tmp_path / "absent.toml")
    assert "not found" in str(exc_info.value).lower()


def test_env_fallback_uses_taste_pipeline_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given an env var pointing to a valid config
    like_lib = tmp_path / "lib"
    like_lib.mkdir()
    body = _minimal_body(like_lib, tmp_path / "data", tmp_path / "cookies.txt")
    config_path = _write_config(tmp_path, body)
    monkeypatch.setenv("TASTE_PIPELINE_CONFIG", str(config_path))

    # When loading with no explicit path
    cfg = load_config()

    # Then the env var is honored
    assert cfg.like_library_dir == like_lib.resolve()


def test_env_fallback_disabled_skips_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given an env var that is set but env_fallback=False
    like_lib = tmp_path / "lib"
    like_lib.mkdir()
    env_path = _write_config(
        tmp_path, _minimal_body(like_lib, tmp_path / "data", tmp_path / "cookies.txt"), name="env.toml"
    )
    monkeypatch.setenv("TASTE_PIPELINE_CONFIG", str(env_path))
    monkeypatch.chdir(tmp_path)
    default_path = _write_config(
        tmp_path,
        _minimal_body(like_lib, tmp_path / "other", tmp_path / "cookies.txt"),
        name="config.toml",
    )

    # When loading with no explicit path and env_fallback=False
    cfg = load_config(env_fallback=False)

    # Then ./config.toml is used (NOT the env var)
    assert cfg.data_dir == (tmp_path / "other").resolve()
    assert default_path.is_file()
