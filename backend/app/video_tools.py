from __future__ import annotations

import re
import subprocess
from pathlib import Path
import math
from uuid import uuid4


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout


def _run_capture_stderr(cmd: list[str]) -> str:
    # ffmpeg log filters (e.g. silencedetect) write to stderr, not stdout.
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stderr


def get_duration_sec(input_path: Path) -> float:
    output = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ]
    )
    duration = float(output.strip())
    if duration <= 0:
        raise RuntimeError("Invalid duration from ffprobe.")
    return duration


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    output = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            str(image_path),
        ]
    )
    width_text, height_text = output.strip().split("x")
    return int(width_text), int(height_text)


def has_audio_stream(input_path: Path) -> bool:
    output = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ]
    )
    return bool(output.strip())


def extract_range(input_path: Path, output_dir: Path, start_sec: float, end_sec: float) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid4()}.mp4"
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ss",
            f"{start_sec:.3f}",
            "-to",
            f"{end_sec:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            str(output_path),
        ]
    )
    return output_path


def remove_segments_and_stitch(
    *,
    input_path: Path,
    output_dir: Path,
    duration_sec: float,
    trim_ranges: list[tuple[float, float]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not trim_ranges:
        return extract_range(input_path, output_dir, 0.0, duration_sec)

    keep_ranges: list[tuple[float, float]] = []
    cursor = 0.0
    for start_sec, end_sec in trim_ranges:
        if start_sec > cursor:
            keep_ranges.append((cursor, start_sec))
        cursor = max(cursor, end_sec)
    if cursor < duration_sec:
        keep_ranges.append((cursor, duration_sec))

    # If everything is trimmed, fail fast.
    if not keep_ranges:
        raise RuntimeError("Cannot export: trim ranges remove the entire video.")

    if len(keep_ranges) == 1:
        start_sec, end_sec = keep_ranges[0]
        return extract_range(input_path, output_dir, start_sec, end_sec)

    output_path = output_dir / f"{uuid4()}.mp4"
    include_audio = has_audio_stream(input_path)

    filters: list[str] = []
    for i, (start_sec, end_sec) in enumerate(keep_ranges):
        filters.append(
            f"[0:v]trim=start={start_sec:.3f}:end={end_sec:.3f},setpts=PTS-STARTPTS[v{i}]"
        )
        if include_audio:
            filters.append(
                f"[0:a]atrim=start={start_sec:.3f}:end={end_sec:.3f},asetpts=PTS-STARTPTS[a{i}]"
            )

    concat_inputs = "".join(
        f"[v{i}]{f'[a{i}]' if include_audio else ''}" for i in range(len(keep_ranges))
    )
    if include_audio:
        filters.append(f"{concat_inputs}concat=n={len(keep_ranges)}:v=1:a=1[v][a]")
    else:
        filters.append(f"{concat_inputs}concat=n={len(keep_ranges)}:v=1:a=0[v]")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
    ]
    if include_audio:
        cmd.extend(["-map", "[a]", "-c:a", "aac"])

    cmd.append(str(output_path))
    _run(cmd)
    return output_path


def _build_atempo_chain(speed: float) -> str:
    # FFmpeg atempo supports [0.5, 2.0] per stage; chain when outside range.
    factors: list[float] = []
    remaining = float(speed)
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={max(0.5, min(2.0, f)):.6f}" for f in factors)


def render_segments_with_speed(
    *,
    input_path: Path,
    output_dir: Path,
    segments: list[tuple[float, float, float]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not segments:
        raise RuntimeError("No segments to render.")

    output_path = output_dir / f"{uuid4()}.mp4"
    include_audio = has_audio_stream(input_path)

    filters: list[str] = []
    for i, (start_sec, end_sec, speed) in enumerate(segments):
        speed_value = max(0.25, float(speed))
        v_chain = (
            f"[0:v]trim=start={start_sec:.3f}:end={end_sec:.3f},"
            f"setpts=PTS-STARTPTS,setpts=PTS/{speed_value:.6f}[v{i}]"
        )
        filters.append(v_chain)

        if include_audio:
            a_chain = (
                f"[0:a]atrim=start={start_sec:.3f}:end={end_sec:.3f},"
                f"asetpts=PTS-STARTPTS,{_build_atempo_chain(speed_value)}[a{i}]"
            )
            filters.append(a_chain)

    concat_inputs = "".join(
        f"[v{i}]{f'[a{i}]' if include_audio else ''}" for i in range(len(segments))
    )
    if include_audio:
        filters.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[v][a]")
    else:
        filters.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=0[v]")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
    ]
    if include_audio:
        cmd.extend(["-map", "[a]", "-c:a", "aac"])

    cmd.append(str(output_path))
    _run(cmd)
    return output_path


def apply_speed_multiplier(
    *,
    input_path: Path,
    output_dir: Path,
    speed_multiplier: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid4()}.mp4"
    multiplier = float(speed_multiplier)
    if multiplier <= 0:
        raise RuntimeError("speed_multiplier must be greater than 0.")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-filter:v",
        f"setpts=PTS/{multiplier}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
    ]

    if has_audio_stream(input_path):
        cmd.extend(["-filter:a", _build_atempo_chain(multiplier), "-c:a", "aac"])

    cmd.append(str(output_path))
    _run(cmd)
    return output_path


def generate_sprite_sheets(
    input_path: Path,
    output_dir: Path,
    *,
    interval_sec: float = 0.25,
    columns: int = 10,
    rows: int = 10,
    thumb_width: int = 320,
) -> dict:
    duration_sec = get_duration_sec(input_path)
    interval_sec = max(0.1, float(interval_sec))
    columns = max(1, int(columns))
    rows = max(1, int(rows))
    thumb_width = max(64, int(thumb_width))

    total_frames = max(1, int(math.floor(duration_sec / interval_sec)) + 1)
    frames_per_sheet = columns * rows
    sheet_count = max(1, int(math.ceil(total_frames / frames_per_sheet)))

    output_dir.mkdir(parents=True, exist_ok=True)
    sheets: list[dict] = []

    for sheet_index in range(sheet_count):
        start_frame = sheet_index * frames_per_sheet
        frame_count = min(frames_per_sheet, total_frames - start_frame)
        start_time_sec = start_frame * interval_sec
        end_time_sec = min(duration_sec, (start_frame + frame_count - 1) * interval_sec)

        image_name = f"sheet_{sheet_index + 1:03d}.png"
        image_path = output_dir / image_name

        filter_graph = (
            f"fps=1/{interval_sec},"
            f"scale={thumb_width}:-1:flags=lanczos,"
            f"tile={columns}x{rows}:nb_frames={frame_count}"
        )

        _run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start_time_sec:.3f}",
                "-i",
                str(input_path),
                "-an",
                "-sn",
                "-dn",
                "-frames:v",
                "1",
                "-vf",
                filter_graph,
                str(image_path),
            ]
        )

        image_width, image_height = get_image_dimensions(image_path)
        tile_width = image_width // columns
        tile_height = image_height // rows

        frames: list[dict] = []
        for i in range(frame_count):
            timestamp = min(duration_sec, (start_frame + i) * interval_sec)
            frames.append(
                {
                    "index": start_frame + i,
                    "timestamp_sec": round(timestamp, 3),
                    "row": i // columns,
                    "col": i % columns,
                }
            )

        sheets.append(
            {
                "sheet_index": sheet_index + 1,
                "image_name": image_name,
                "image_width": image_width,
                "image_height": image_height,
                "tile_width": tile_width,
                "tile_height": tile_height,
                "start_time_sec": round(start_time_sec, 3),
                "end_time_sec": round(end_time_sec, 3),
                "frames": frames,
            }
        )

    return {
        "duration_sec": round(duration_sec, 3),
        "interval_sec": interval_sec,
        "columns": columns,
        "rows": rows,
        "total_frames": total_frames,
        "sheets": sheets,
    }


def detect_silence(
    input_path: Path,
    *,
    noise_db: float = -30.0,
    min_duration_sec: float = 0.5,
) -> list[tuple[float, float]]:
    """Detect silent audio ranges via FFmpeg's silencedetect filter. Returns
    (start_sec, end_sec) pairs. If the file ends while still silent, the final
    range is clipped to the file's duration."""
    stderr = _run_capture_stderr(
        [
            "ffmpeg",
            "-i",
            str(input_path),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={min_duration_sec}",
            "-f",
            "null",
            "-",
        ]
    )
    starts = [float(m) for m in re.findall(r"silence_start:\s*(-?[\d.]+)", stderr)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*(-?[\d.]+)", stderr)]
    pairs = list(zip(starts, ends))
    if len(starts) > len(ends):
        pairs.append((starts[-1], get_duration_sec(input_path)))
    return pairs


def detect_scene_changes(
    input_path: Path,
    *,
    threshold: float = 0.3,
    max_results: int = 6,
) -> list[float]:
    """Detect scene-cut timestamps via FFmpeg's scene-change score (ADR-0002/L3).
    Higher threshold = fewer, more confident cuts. Returns up to max_results
    timestamps in chronological order."""
    stderr = _run_capture_stderr(
        [
            "ffmpeg",
            "-i",
            str(input_path),
            "-vf",
            f"select='gt(scene,{threshold})',showinfo",
            "-f",
            "null",
            "-",
        ]
    )
    timestamps = [float(m) for m in re.findall(r"pts_time:([\d.]+)", stderr)]
    return timestamps[:max_results]


def extract_thumbnail(
    input_path: Path, output_dir: Path, timestamp_sec: float, thumb_width: int = 256
) -> Path:
    """Extract a single frame at timestamp_sec as a PNG thumbnail."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid4()}.png"
    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{max(0.0, timestamp_sec):.3f}",
            "-i",
            str(input_path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={thumb_width}:-1:flags=lanczos",
            str(output_path),
        ]
    )
    return output_path
