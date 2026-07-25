from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from ..video_tools import get_duration_sec


def validate_sprite_params(interval_sec: float, columns: int, rows: int) -> None:
    if interval_sec <= 0:
        raise ValueError("interval_sec must be greater than 0.")
    if columns <= 0 or rows <= 0:
        raise ValueError("columns and rows must be greater than 0.")


async def save_upload_file(
    *,
    file: UploadFile,
    upload_dir: Path,
    max_file_size_mb: int | None = None,
) -> Path:
    # ponytail: whole file buffered in memory, bounded by max_file_size_mb (500MB
    # default). Fine at the ~15-20 min v1 ceiling (ADR-0006); switch to a streamed
    # chunked write if the ceiling grows toward full long-form.
    body = await file.read()
    if max_file_size_mb is not None and len(body) > max_file_size_mb * 1024 * 1024:
        raise ValueError(f"File exceeds {max_file_size_mb} MB")

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    save_path = upload_dir / f"{uuid4()}{suffix}"
    save_path.write_bytes(body)
    return save_path


def probe_duration_or_cleanup(input_path: Path) -> float:
    try:
        return get_duration_sec(input_path)
    except Exception as exc:
        input_path.unlink(missing_ok=True)
        raise ValueError(f"Invalid media file: {exc}") from exc


def find_persisted_upload(uploads_dir: Path, sprite_job_id: str) -> Path | None:
    matches = list(uploads_dir.glob(f"{sprite_job_id}.*"))
    return matches[0] if matches else None
