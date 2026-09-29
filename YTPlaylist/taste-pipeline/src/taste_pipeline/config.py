"""TOML configuration with eager validation.

The pipeline loads ``config.toml`` (path chosen from ``--config`` CLI
argument, ``TASTE_PIPELINE_CONFIG`` env, or ``./config.toml``), parses it
with the stdlib ``tomllib``, and builds an immutable :class:`Config`.
Every invalid key is reported in one :class:`ConfigError`; path fields
are resolved against the config file's directory; the data directory
and its standard subfolders are created on load when missing.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

_DATA_SUBDIRS: Final[tuple[str, ...]] = (
    "inbox",
    "keep",
    "review",
    "skip",
    "dislike",
    "index",
    "runs",
)
_DOWNLOAD_ARCHIVE_NAME: Final[str] = "yt-dlp-archive.txt"
_WITH_MIN_BOUND: Final[int] = 1
_WITH_MAX_BOUND: Final[int] = 2
_DEVICE_CHOICES: Final[str] = "auto, cpu, cuda, mps"
_ALLOWED_DEVICES: Final[frozenset[str]] = frozenset(_DEVICE_CHOICES.split(", "))

# Per-field validation specs: (kind, *bounds). kind ∈ path_req, path_opt, str, opt_str, int, float, opt_float.
_FIELD_SPECS: Final[dict[str, tuple]] = {
    "like_library_dir": ("path_req",),
    "data_dir": ("path_req",),
    "cookie_file": ("path_req",),
    "download_archive": ("path_opt",),
    "proxy": ("opt_str",),
    "feed_window_days": ("int", 1),
    "max_feed_items": ("int", 1),
    "max_metadata_fetch": ("int", 1),
    "chunk_seconds": ("float", 0.0),
    "min_chunk_seconds": ("float", 0.0),
    "sample_rate": ("int", 1),
    "keep_threshold": ("opt_float", 0.0, 1.0),
    "skip_threshold": ("opt_float", 0.0, 1.0),
    "min_dislikes_for_classifier": ("int", 0),
    "device": ("str",),
    "web_host": ("str",),
    "web_port": ("int", 1, 65535),
}


class ConfigError(Exception):
    """Raised when the config file is missing, malformed, or has invalid fields; lists every invalid key."""


@dataclass(frozen=True, slots=True)
class Config:
    """Validated pipeline configuration.

    Path fields are absolute ``pathlib.Path`` resolved against the
    directory containing the loaded config file.
    """

    like_library_dir: Path
    data_dir: Path
    cookie_file: Path
    download_archive: Path
    proxy: str | None = None
    feed_window_days: int = 7
    max_feed_items: int = 500
    max_metadata_fetch: int = 150
    chunk_seconds: float = 10.0
    min_chunk_seconds: float = 3.0
    sample_rate: int = 48000
    keep_threshold: float | None = None
    skip_threshold: float | None = None
    min_dislikes_for_classifier: int = 30
    device: str = "auto"
    web_host: str = "127.0.0.1"
    web_port: int = 8741


def _typename(value: object) -> str:
    """User-friendly type name; bool is rendered as 'bool' (not 'int')."""
    return "bool" if isinstance(value, bool) else type(value).__name__


def _coerce_str(name: str, value: object, errors: list[str]) -> str | None:
    """Validate a TOML value as a non-empty string. None + error on failure."""
    if not isinstance(value, str):
        errors.append(f"{name}: must be a string (got {_typename(value)})")
        return None
    if not value:
        errors.append(f"{name}: must be a non-empty string")
        return None
    return value


def _coerce_path(name: str, value: object, base_dir: Path, errors: list[str]) -> Path | None:
    """Validate a TOML value as a non-empty string path, then resolve it against ``base_dir``."""
    text = _coerce_str(name, value, errors)
    if text is None:
        return None
    expanded = Path(text).expanduser()
    if expanded.is_absolute():
        return expanded.resolve(strict=False)
    return (base_dir / expanded).resolve(strict=False)


def _coerce_int(
    name: str,
    value: object,
    errors: list[str],
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    """Validate a TOML value as an int (rejecting bool); apply optional bounds."""
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{name}: must be an int (got {_typename(value)})")
        return None
    if minimum is not None and value < minimum:
        errors.append(f"{name}: must be >= {minimum} (got {value})")
        return None
    if maximum is not None and value > maximum:
        errors.append(f"{name}: must be <= {maximum} (got {value})")
        return None
    return value


def _coerce_float(
    name: str,
    value: object,
    errors: list[str],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    """Validate a TOML value as a number (int or float, not bool); apply optional bounds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{name}: must be a number (got {_typename(value)})")
        return None
    result = float(value)
    if minimum is not None and result < minimum:
        errors.append(f"{name}: must be >= {minimum} (got {result})")
        return None
    if maximum is not None and result > maximum:
        errors.append(f"{name}: must be <= {maximum} (got {result})")
        return None
    return result


def _coerce_optional_float(
    name: str,
    value: object,
    errors: list[str],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    """Validate a TOML value as a number or null; null passes through."""
    if value is None:
        return None
    return _coerce_float(name, value, errors, minimum=minimum, maximum=maximum)


def _coerce_field(name: str, value: object, spec: tuple, base_dir: Path, errors: list[str]) -> object | None:
    """Dispatch one field by its spec. Returns the coerced value or ``None`` (error appended)."""
    kind = spec[0]
    if kind in {"str", "opt_str"}:
        if kind == "opt_str" and value is None:
            return None
        return _coerce_str(name, value, errors)
    if kind == "path_req":
        return _coerce_path(name, value, base_dir, errors)
    if kind == "path_opt":
        if value is None:
            return None
        return _coerce_path(name, value, base_dir, errors)
    coerce = {
        "int": _coerce_int,
        "float": _coerce_float,
        "opt_float": _coerce_optional_float,
    }[kind]
    spec_len = len(spec)
    lo: int | float | None = spec[1] if spec_len > _WITH_MIN_BOUND else None
    hi: int | float | None = spec[2] if spec_len > _WITH_MAX_BOUND else None
    return coerce(name, value, errors, minimum=lo, maximum=hi)


def _extract_all_fields(
    raw: dict[str, object], base_dir: Path, values: dict[str, object], errors: list[str]
) -> None:
    """Coerce each spec entry from ``raw`` (when present) into ``values``.

    Missing required path fields get a 'required' error; missing optional
    fields fall back to the dataclass defaults.
    """
    for name, spec in _FIELD_SPECS.items():
        if name not in raw:
            if spec[0] == "path_req":
                errors.append(f"{name}: required")
            continue
        coerced = _coerce_field(name, raw[name], spec, base_dir, errors)
        if coerced is not None:
            values[name] = coerced


def _validate(raw: dict[str, object], config_path: Path) -> Config:
    """Validate ``raw`` and return a ``Config``; raise ``ConfigError`` with every error aggregated."""
    errors: list[str] = []
    base_dir = config_path.parent.resolve(strict=False)
    values: dict[str, object] = {}

    valid_keys = set(_FIELD_SPECS)
    errors.extend(f"{key}: unknown key" for key in sorted(set(raw) - valid_keys))
    _extract_all_fields(raw, base_dir, values, errors)

    if "download_archive" not in values and "data_dir" in values:
        data_dir = cast("Path", values["data_dir"])
        values["download_archive"] = data_dir / _DOWNLOAD_ARCHIVE_NAME

    chunk, min_chunk = values.get("chunk_seconds"), values.get("min_chunk_seconds")
    if isinstance(chunk, float) and isinstance(min_chunk, float) and min_chunk > chunk:
        errors.append(f"min_chunk_seconds ({min_chunk}) must not exceed chunk_seconds ({chunk})")
    keep, skip = values.get("keep_threshold"), values.get("skip_threshold")
    if isinstance(keep, float) and isinstance(skip, float) and skip >= keep:
        errors.append(f"skip_threshold ({skip}) must be strictly less than keep_threshold ({keep})")
    device = values.get("device")
    if isinstance(device, str) and device not in _ALLOWED_DEVICES:
        errors.append(f"device: must be one of {_DEVICE_CHOICES} (got '{device}')")
    if errors:
        bullets = "\n  - ".join(errors)
        message = f"invalid config {config_path}:\n  - {bullets}"
        raise ConfigError(message)

    like_library_dir = cast("Path", values["like_library_dir"])
    data_dir = cast("Path", values["data_dir"])
    if not like_library_dir.is_dir():
        message = (
            f"invalid config {config_path}: like_library_dir={like_library_dir} "
            "does not exist or is not a directory"
        )
        raise ConfigError(message)
    data_dir.mkdir(parents=True, exist_ok=True)
    for sub in _DATA_SUBDIRS:
        (data_dir / sub).mkdir(parents=True, exist_ok=True)

    # Every field was coerced + validated above, so the dynamic kwargs match
    # the dataclass signature; the type checker cannot see through **values.
    return Config(**values)  # pyright: ignore[reportArgumentType]


def load_config(
    path: str | os.PathLike[str] | None = None,
    env_fallback: bool = True,
) -> Config:
    """Load and validate a TOML configuration file.

    Path resolution: explicit ``path`` wins; else ``$TASTE_PIPELINE_CONFIG``
    when ``env_fallback``; else ``./config.toml``.

    Returns:
        A validated :class:`Config` with all path fields resolved and
        the data directory created.

    Raises:
        ConfigError: The file is missing, unreadable, malformed, or
            has invalid fields. The message lists every invalid key.
    """
    if path is None and env_fallback:
        env_path = os.environ.get("TASTE_PIPELINE_CONFIG")
        if env_path:
            path = env_path
    if path is None:
        path = "config.toml"
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        message = f"config file not found: {config_path}"
        raise ConfigError(message)
    try:
        with config_path.open("rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        message = f"invalid TOML in {config_path}: {exc}"
        raise ConfigError(message) from exc
    if not isinstance(raw, dict):
        message = f"invalid config {config_path}: top-level must be a table, got {_typename(raw)}"
        raise ConfigError(message)
    return _validate(raw, config_path)
