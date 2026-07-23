from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from .gemini_agent import suggest_cuts_from_sprites
from .schemas import (
    ExportResponse,
    SuggestCutsRequest,
    SuggestCutsResponse,
    SpriteAnalysisResponse,
    TokenEstimateRequest,
    TokenEstimateResponse,
    TrimRange,
)
from .services.media_service import (
    probe_duration_or_cleanup,
    save_upload_file,
    validate_sprite_params,
)
from .services.token_service import estimate_tokens
from .validators import validate_trim
from .video_tools import (
    apply_speed_multiplier,
    extract_range,
    generate_sprite_sheets,
    get_duration_sec,
    remove_segments_and_stitch,
    render_segments_with_speed,
)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_ROOT / ".env")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(levelname)s:%(name)s:%(message)s",
    )
logging.getLogger("app").setLevel(LOG_LEVEL)
logging.getLogger("app.gemini_agent").setLevel(LOG_LEVEL)
MEDIA_ROOT = (BACKEND_ROOT / os.getenv("MEDIA_ROOT", "media")).resolve()
MAX_VIDEO_DURATION_SEC = float(os.getenv("MAX_VIDEO_DURATION_SEC", "1200"))  # 20 min, ADR-0006
OUTPUT_TTL_MIN = float(os.getenv("OUTPUT_TTL_MIN", "60"))
UPLOAD_DIR = MEDIA_ROOT / "uploads"
OUTPUT_DIR = MEDIA_ROOT / "outputs"
SPRITES_DIR = MEDIA_ROOT / "sprites"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPRITES_DIR.mkdir(parents=True, exist_ok=True)


def _allowed_origins() -> list[str]:
    origins = {"http://localhost:3000", "http://127.0.0.1:3000"}
    env_origins = os.getenv("CORS_ORIGINS", "")
    if env_origins.strip():
        for origin in env_origins.split(","):
            cleaned = origin.strip().rstrip("/")
            if cleaned:
                origins.add(cleaned)
    vercel_frontend_url = os.getenv("VERCEL_FRONTEND_URL", "").strip().rstrip("/")
    if vercel_frontend_url:
        origins.add(vercel_frontend_url)
    return sorted(origins)


def _sweep_old_outputs() -> None:
    # ponytail: opportunistic sweep-on-export, not a background timer — survives
    # the scale-to-zero host since it only runs when a request is actually in flight.
    if OUTPUT_TTL_MIN <= 0:
        return
    cutoff = time.time() - OUTPUT_TTL_MIN * 60
    for path in OUTPUT_DIR.glob("*"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            continue


def _enforce_max_duration(duration_sec: float) -> None:
    if duration_sec > MAX_VIDEO_DURATION_SEC:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Video duration {duration_sec:.2f}s exceeds maximum "
                f"{MAX_VIDEO_DURATION_SEC:.2f}s."
            ),
        )


def _merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda x: x[0])
    merged = [ordered[0]]
    for start_sec, end_sec in ordered[1:]:
        last_start, last_end = merged[-1]
        if start_sec <= last_end:
            merged[-1] = (last_start, max(last_end, end_sec))
        else:
            merged.append((start_sec, end_sec))
    return merged


def _build_speed_segments(
    *,
    duration_sec: float,
    trim_ranges: list[tuple[float, float]],
    speed_ranges: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    keep_ranges: list[tuple[float, float]] = []
    cursor = 0.0
    for trim_start, trim_end in trim_ranges:
        if trim_start > cursor:
            keep_ranges.append((cursor, trim_start))
        cursor = max(cursor, trim_end)
    if cursor < duration_sec:
        keep_ranges.append((cursor, duration_sec))

    if not trim_ranges:
        keep_ranges = [(0.0, duration_sec)]

    if not keep_ranges:
        return []

    if not speed_ranges:
        return [(start, end, 1.0) for start, end in keep_ranges]

    speed_ranges_sorted = sorted(speed_ranges, key=lambda x: x[0])
    for i in range(1, len(speed_ranges_sorted)):
        prev = speed_ranges_sorted[i - 1]
        current = speed_ranges_sorted[i]
        if current[0] < prev[1]:
            raise HTTPException(status_code=400, detail="Overlapping speed ranges are not supported.")

    segments: list[tuple[float, float, float]] = []
    for keep_start, keep_end in keep_ranges:
        cursor = keep_start
        for speed_start, speed_end, speed_value in speed_ranges_sorted:
            if speed_end <= keep_start:
                continue
            if speed_start >= keep_end:
                break

            overlap_start = max(keep_start, speed_start)
            overlap_end = min(keep_end, speed_end)
            if overlap_end <= overlap_start:
                continue

            if overlap_start > cursor:
                segments.append((cursor, overlap_start, 1.0))
            segments.append((overlap_start, overlap_end, speed_value))
            cursor = overlap_end

        if cursor < keep_end:
            segments.append((cursor, keep_end, 1.0))

    return [(s, e, sp) for s, e, sp in segments if e - s > 0.01]


app = FastAPI(title="Video Editor Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/media/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/media/sprites", StaticFiles(directory=str(SPRITES_DIR)), name="sprites")

API_KEY = os.getenv("API_KEY", "").strip()


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    # ponytail: single shared secret (self-host, single-user); /health and /media/*
    # stay open since <video>/<img> tags can't send custom headers. Add per-user
    # auth if this ever goes multi-tenant.
    open_path = request.url.path == "/health" or request.url.path.startswith("/media/")
    if API_KEY and request.method != "OPTIONS" and not open_path:
        if request.headers.get("x-api-key") != API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key."})
    return await call_next(request)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/analyze/sprites", response_model=SpriteAnalysisResponse)
async def analyze_sprites(
    file: UploadFile = File(...),
    interval_sec: float = Form(0.25),
    columns: int = Form(10),
    rows: int = Form(10),
    thumb_width: int = Form(320),
) -> SpriteAnalysisResponse:
    try:
        validate_sprite_params(interval_sec, columns, rows)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    upload_path = await save_upload_file(file=file, upload_dir=UPLOAD_DIR)
    try:
        duration_sec = await asyncio.to_thread(probe_duration_or_cleanup, upload_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _enforce_max_duration(duration_sec)
    persist_sprites = os.getenv("SPRITE_PERSIST", "false").strip().lower() == "true"
    sprite_job_id = str(uuid4())

    try:
        if persist_sprites:
            sprite_output_dir = SPRITES_DIR / sprite_job_id
            analysis = await asyncio.to_thread(
                generate_sprite_sheets,
                input_path=upload_path,
                output_dir=sprite_output_dir,
                interval_sec=interval_sec,
                columns=columns,
                rows=rows,
                thumb_width=thumb_width,
            )
            sheets = []
            for sheet in analysis["sheets"]:
                sheets.append(
                    {
                        "sheet_index": sheet["sheet_index"],
                        "image_url": f"/media/sprites/{sprite_job_id}/{sheet['image_name']}",
                        "image_width": sheet["image_width"],
                        "image_height": sheet["image_height"],
                        "tile_width": sheet["tile_width"],
                        "tile_height": sheet["tile_height"],
                        "start_time_sec": sheet["start_time_sec"],
                        "end_time_sec": sheet["end_time_sec"],
                        "frames": sheet["frames"],
                    }
                )
        else:
            with tempfile.TemporaryDirectory(prefix="sprite_job_") as temp_dir:
                analysis = await asyncio.to_thread(
                    generate_sprite_sheets,
                    input_path=upload_path,
                    output_dir=Path(temp_dir),
                    interval_sec=interval_sec,
                    columns=columns,
                    rows=rows,
                    thumb_width=thumb_width,
                )
            # No persisted files in non-persistent mode.
            sheets = []
            for sheet in analysis["sheets"]:
                sheets.append(
                    {
                        "sheet_index": sheet["sheet_index"],
                        "image_url": "",
                        "image_width": sheet["image_width"],
                        "image_height": sheet["image_height"],
                        "tile_width": sheet["tile_width"],
                        "tile_height": sheet["tile_height"],
                        "start_time_sec": sheet["start_time_sec"],
                        "end_time_sec": sheet["end_time_sec"],
                        "frames": sheet["frames"],
                    }
                )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sprite analysis failed: {exc}") from exc
    finally:
        upload_path.unlink(missing_ok=True)

    return SpriteAnalysisResponse(
        duration_sec=analysis["duration_sec"],
        interval_sec=analysis["interval_sec"],
        columns=analysis["columns"],
        rows=analysis["rows"],
        total_frames=analysis["total_frames"],
        sheets=sheets,
    )


@app.post("/analyze/token-estimate", response_model=TokenEstimateResponse)
def analyze_token_estimate(payload: TokenEstimateRequest) -> TokenEstimateResponse:
    return estimate_tokens(
        duration_sec=payload.duration_sec,
        interval_sec=payload.interval_sec,
        columns=payload.columns,
        rows=payload.rows,
        thumb_width=payload.thumb_width,
    )


@app.post("/analyze/token-estimate-from-file", response_model=TokenEstimateResponse)
async def analyze_token_estimate_from_file(
    file: UploadFile = File(...),
    interval_sec: float = Form(0.25),
    columns: int = Form(8),
    rows: int = Form(8),
    thumb_width: int = Form(256),
) -> TokenEstimateResponse:
    try:
        validate_sprite_params(interval_sec, columns, rows)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    save_path = await save_upload_file(file=file, upload_dir=UPLOAD_DIR)

    try:
        try:
            duration_sec = await asyncio.to_thread(probe_duration_or_cleanup, save_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _enforce_max_duration(duration_sec)

        return estimate_tokens(
            duration_sec=duration_sec,
            interval_sec=interval_sec,
            columns=columns,
            rows=rows,
            thumb_width=thumb_width,
        )
    finally:
        save_path.unlink(missing_ok=True)


@app.post("/ai/suggest-cuts-from-sprites", response_model=SuggestCutsResponse)
async def ai_suggest_cuts_from_sprites(payload: SuggestCutsRequest) -> SuggestCutsResponse:
    try:
        result = await suggest_cuts_from_sprites(
            prompt=payload.prompt,
            duration_sec=payload.duration_sec,
            sprite_interval_sec=payload.sprite_interval_sec,
            total_frames=payload.total_frames,
            sheets_count=payload.sheets_count,
            chat_history=[item.model_dump() for item in payload.chat_history],
            conversation_summary=payload.conversation_summary,
            trim_ranges=[item.model_dump() for item in payload.trim_ranges],
            speed_ranges=[item.model_dump() for item in payload.speed_ranges],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SuggestCutsResponse(
        suggestions=result["suggestions"],
        model=result["model"],
        strategy=result["strategy"],
    )


@app.post("/export/from-file", response_model=ExportResponse)
async def export_from_file(
    file: UploadFile = File(...),
    trim_ranges: str = Form(default="[]"),
    speed_ranges: str = Form(default="[]"),
    speed_multiplier: Optional[float] = Form(default=None),
    speed_factor: Optional[float] = Form(default=None),
    speed: Optional[str] = Form(default=None),
) -> ExportResponse:
    await asyncio.to_thread(_sweep_old_outputs)
    max_mb = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
    try:
        input_path = await save_upload_file(
            file=file, upload_dir=UPLOAD_DIR, max_file_size_mb=max_mb
        )
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    try:
        duration_sec = await asyncio.to_thread(probe_duration_or_cleanup, input_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _enforce_max_duration(duration_sec)

    try:
        raw_ranges = json.loads(trim_ranges)
        parsed_ranges = [TrimRange.model_validate(item) for item in raw_ranges]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid trim_ranges JSON: {exc}") from exc

    normalized_ranges: list[tuple[float, float]] = []
    for item in parsed_ranges:
        start_sec = float(min(item.start, item.end))
        end_sec = float(max(item.start, item.end))
        validate_trim(start_sec, end_sec, duration_sec)
        normalized_ranges.append((start_sec, end_sec))

    normalized_ranges.sort(key=lambda x: x[0])
    merged_ranges = _merge_ranges(normalized_ranges)

    try:
        raw_speed_ranges = json.loads(speed_ranges)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid speed_ranges JSON: {exc}") from exc

    normalized_speed_ranges: list[tuple[float, float, float]] = []
    for item in raw_speed_ranges:
        start_sec = float(min(item["start"], item["end"]))
        end_sec = float(max(item["start"], item["end"]))
        speed_value = float(item.get("speed", 1.0))
        validate_trim(start_sec, end_sec, duration_sec)
        if speed_value <= 0:
            raise HTTPException(status_code=400, detail="Speed must be greater than 0.")
        normalized_speed_ranges.append((start_sec, end_sec, speed_value))

    if merged_ranges and merged_ranges[0][0] <= 0 and merged_ranges[-1][1] >= duration_sec:
        # Entire timeline removed after merge.
        only_removed = len(merged_ranges) == 1 and merged_ranges[0][0] <= 0 and merged_ranges[0][1] >= duration_sec
        if only_removed:
            raise HTTPException(status_code=400, detail="Cannot remove the entire video range.")

    selected_speed = 1.0
    if speed_multiplier is not None:
        selected_speed = float(speed_multiplier)
    elif speed_factor is not None:
        selected_speed = float(speed_factor)
    elif speed is not None:
        speed_value = str(speed).strip().lower().removesuffix("x")
        try:
            selected_speed = float(speed_value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid speed value.") from exc

    if selected_speed not in {1.0, 2.0}:
        raise HTTPException(status_code=400, detail="Only 1x and 2x are supported in v0.")

    try:
        try:
            speed_segments = _build_speed_segments(
                duration_sec=duration_sec,
                trim_ranges=merged_ranges,
                speed_ranges=normalized_speed_ranges,
            )
            if speed_segments and any(abs(seg[2] - 1.0) > 1e-6 for seg in speed_segments):
                output_path = await asyncio.to_thread(
                    render_segments_with_speed,
                    input_path=input_path,
                    output_dir=OUTPUT_DIR,
                    segments=speed_segments,
                )
            else:
                if merged_ranges:
                    output_path = await asyncio.to_thread(
                        remove_segments_and_stitch,
                        input_path=input_path,
                        output_dir=OUTPUT_DIR,
                        duration_sec=duration_sec,
                        trim_ranges=merged_ranges,
                    )
                else:
                    # No trims: produce a normal export copy by re-encoding the full source range.
                    output_path = await asyncio.to_thread(
                        extract_range,
                        input_path=input_path,
                        output_dir=OUTPUT_DIR,
                        start_sec=0.0,
                        end_sec=duration_sec,
                    )

            if selected_speed > 1.0 and not normalized_speed_ranges:
                speed_output_path = await asyncio.to_thread(
                    apply_speed_multiplier,
                    input_path=output_path,
                    output_dir=OUTPUT_DIR,
                    speed_multiplier=selected_speed,
                )
                output_path.unlink(missing_ok=True)
                output_path = speed_output_path
            # Sanity check output can be probed.
            _ = await asyncio.to_thread(get_duration_sec, output_path)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Export failed: {exc}") from exc
    finally:
        input_path.unlink(missing_ok=True)

    return ExportResponse(
        output_url=f"/media/outputs/{output_path.name}",
        output_name=output_path.name,
        removed_ranges_count=len(merged_ranges),
    )


@app.exception_handler(Exception)
async def fallback_exception_handler(_request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
