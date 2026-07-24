from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional
from uuid import uuid4

import httpx

from .validators import parse_time_like, validate_trim
from .video_tools import detect_silence

logger = logging.getLogger(__name__)
MAX_CONTEXT_TURNS = 10
MAX_SUMMARY_CHARS = 500
# ponytail: fixed cap keeps the coarse vision pass's cost flat regardless of video
# duration (ADR-0002/ADR-0006) — evenly subsample instead of sending every sheet.
MAX_SPRITE_IMAGES = 6
GEMINI_MODEL = "gemini-2.5-flash"  # gemini-2.0-flash was shut down 2026-06-01
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
_SILENCE_KEYWORDS = re.compile(
    r"\b(dead\s*air|silence|silent|pause[s]?|awkward gap[s]?)\b", re.IGNORECASE
)


def _wants_silence_removal(prompt: str) -> bool:
    return bool(_SILENCE_KEYWORDS.search(prompt))


def _find_persisted_upload(uploads_dir: Path, sprite_job_id: str) -> Optional[Path]:
    matches = list(uploads_dir.glob(f"{sprite_job_id}.*"))
    return matches[0] if matches else None


def _silence_proposals(silences: list[tuple[float, float]]) -> list[dict]:
    return [
        {
            "action": "trim_video",
            "operation": "remove_segment",
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "reason": f"Detected {round(end - start, 2)}s of silence via audio analysis.",
            "confidence": 0.9,
            "speed_multiplier": None,
        }
        for start, end in silences
    ]


def _select_sprite_files(sprites_dir: Path, sprite_job_id: str, limit: int) -> list[Path]:
    job_dir = sprites_dir / sprite_job_id
    if not job_dir.is_dir():
        return []
    files = sorted(job_dir.glob("sheet_*.png"))
    if len(files) <= limit:
        return files
    step = len(files) / limit
    return [files[int(i * step)] for i in range(limit)]


def _encode_sprite_images(files: list[Path]) -> list[dict]:
    parts = []
    for path in files:
        try:
            data = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            continue
        parts.append({"inline_data": {"mime_type": "image/png", "data": data}})
    return parts


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object in model output.")
    return json.loads(text[start : end + 1])


def _fallback_suggest_cuts(prompt: str, duration_sec: float) -> list[dict]:
    lowered = prompt.lower()
    matches = re.findall(r"(?:from|between)\s+([0-9:.]+)\s+(?:to|and|-)\s+([0-9:.]+)", lowered)
    speed_hint = bool(re.search(r"\b(speed\s*up|faster|fast-forward|accelerat(?:e|ed|ing)|\d+(?:\.\d+)?x)\b", lowered))
    multiplier_match = re.search(r"(\d+(?:\.\d+)?)\s*x", lowered)
    speed_multiplier = float(multiplier_match.group(1)) if multiplier_match else 2.0
    if matches:
        suggestions = []
        for start_raw, end_raw in matches:
            start_sec = parse_time_like(start_raw)
            end_sec = parse_time_like(end_raw)
            if 0 <= start_sec < end_sec <= duration_sec:
                if speed_hint:
                    suggestions.append(
                        {
                            "action": "speed_video",
                            "operation": "apply_speed_range",
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "speed_multiplier": speed_multiplier,
                            "reason": "Parsed explicit speed range from prompt.",
                            "confidence": 0.9,
                        }
                    )
                    continue
                suggestions.append(
                    {
                        "action": "trim_video",
                        "operation": "remove_segment",
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "reason": "Parsed explicit range from prompt.",
                        "confidence": 0.9,
                    }
                )
        if suggestions:
            return suggestions

    if "recommend" in lowered or "suggest" in lowered:
        segment = max(0.5, duration_sec * 0.08)
        points = [duration_sec * 0.22, duration_sec * 0.5, duration_sec * 0.78]
        suggestions = []
        for p in points:
            start_sec = max(0.0, p - segment / 2)
            end_sec = min(duration_sec, start_sec + segment)
            if end_sec - start_sec >= 0.3:
                suggestions.append(
                    {
                        "action": "trim_video",
                        "operation": "remove_segment",
                        "start_sec": round(start_sec, 3),
                        "end_sec": round(end_sec, 3),
                        "reason": "Fallback recommendation window.",
                        "confidence": 0.45,
                    }
                )
        return suggestions[:3]

    return []


def _normalize_suggestions(raw_suggestions: list, duration_sec: float) -> list[dict]:
    # ponytail: reuse validators.validate_trim as the single source of truth for
    # range validity instead of re-deriving the same 0<=start<end<=duration check.
    normalized: list[dict] = []
    for item in raw_suggestions:
        try:
            start_sec = parse_time_like(item["start_sec"])
            end_sec = parse_time_like(item["end_sec"])
            validate_trim(start_sec, end_sec, duration_sec)
        except Exception:
            continue
        action = str(item.get("action", "trim_video"))
        if action not in {"trim_video", "speed_video"}:
            action = "trim_video"
        operation_default = "apply_speed_range" if action == "speed_video" else "remove_segment"
        operation = str(item.get("operation", operation_default))
        if operation not in {"remove_segment", "extract_range", "apply_speed_range"}:
            operation = operation_default
        confidence_raw = item.get("confidence", 0.5)
        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 0.5
        speed_multiplier = None
        if action == "speed_video":
            raw_multiplier = item.get("speed_multiplier", 2.0)
            try:
                speed_multiplier = float(raw_multiplier)
            except Exception:
                speed_multiplier = 2.0
            speed_multiplier = max(0.25, min(16.0, speed_multiplier))
        normalized.append(
            {
                "action": action,
                "operation": operation,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "reason": str(item.get("reason", "Model suggestion")),
                "confidence": max(0.0, min(1.0, confidence)),
                "speed_multiplier": speed_multiplier,
            }
        )
    return normalized


async def _call_gemini(api_key: str, parts: list[dict]) -> dict:
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{GEMINI_URL}?key={api_key}", json=payload)
        response.raise_for_status()
        data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    logger.info("GEMINI_RAW_PLAN_RESPONSE %s", text)
    return _extract_json(text)


async def plan_edits(
    *,
    prompt: str,
    duration_sec: float,
    sprite_interval_sec: float,
    total_frames: int,
    sheets_count: int,
    chat_history: Optional[list[dict]] = None,
    conversation_summary: Optional[str] = None,
    trim_ranges: Optional[list[dict]] = None,
    speed_ranges: Optional[list[dict]] = None,
    sprite_job_id: Optional[str] = None,
    sprites_dir: Optional[Path] = None,
    uploads_dir: Optional[Path] = None,
) -> dict:
    """Plan -> validate -> propose (ADR-0003). One bounded self-correction retry
    if the model's first attempt yields zero valid proposals; falls back to the
    regex heuristic only if that retry also fails. Silence/dead-air detection
    (X5) is a deterministic FFmpeg tool that runs independently of Gemini and
    merges its proposals in regardless of which path produced the rest."""
    plan_id = str(uuid4())
    api_key = os.getenv("GEMINI_API_KEY")
    sprite_files: list[Path] = []
    if api_key and sprite_job_id and sprites_dir:
        sprite_files = await asyncio.to_thread(
            _select_sprite_files, sprites_dir, sprite_job_id, MAX_SPRITE_IMAGES
        )

    silence_suggestions: list[dict] = []
    if _wants_silence_removal(prompt) and sprite_job_id and uploads_dir:
        source_path = await asyncio.to_thread(_find_persisted_upload, uploads_dir, sprite_job_id)
        if source_path is not None:
            try:
                silences = await asyncio.to_thread(detect_silence, source_path)
                silence_suggestions = _silence_proposals(silences)
            except Exception:
                logger.exception("SILENCE_DETECTION_FAILED")

    logger.info(
        "PLAN_EDITS_REQUEST %s",
        {
            "plan_id": plan_id,
            "duration_sec": duration_sec,
            "sprite_interval_sec": sprite_interval_sec,
            "total_frames": total_frames,
            "sheets_count": sheets_count,
            "prompt": prompt,
            "chat_history_count": len(chat_history or []),
            "trim_ranges_count": len(trim_ranges or []),
            "speed_ranges_count": len(speed_ranges or []),
            "sprite_images_attached": len(sprite_files),
            "silence_proposals": len(silence_suggestions),
        },
    )
    if not api_key:
        fallback_suggestions = _fallback_suggest_cuts(prompt, duration_sec) + silence_suggestions
        result = {
            "plan_id": plan_id,
            "model": "fallback",
            "strategy": "rule-based",
            "reasoning": "No Gemini API key configured; parsed the prompt with regex heuristics.",
            "proposals": [{**item, "id": str(uuid4())} for item in fallback_suggestions],
        }
        logger.info("PLAN_EDITS_FALLBACK_RESPONSE %s", result)
        return result

    instructions = (
        "You are an editing planner. Return strict JSON only.\n"
        "Schema: {\"reasoning\":string,\"suggestions\":[{\"action\":\"trim_video|speed_video\",\"operation\":\"remove_segment|extract_range|apply_speed_range\",\"start_sec\":number,\"end_sec\":number,\"speed_multiplier\":number,\"reason\":string,\"confidence\":number}]}\n"
        f"Video duration: {duration_sec:.3f}s\n"
        f"Sprite analysis summary: interval={sprite_interval_sec}s, total_frames={total_frames}, sheets={sheets_count}\n"
        "Rules:\n"
        "- reasoning is one short sentence summarizing your overall plan.\n"
        "- Produce 0 to 8 suggestions.\n"
        "- Each suggestion must satisfy 0 <= start_sec < end_sec <= duration.\n"
        "- Confidence range 0..1\n"
        "- If prompt asks recommendation, infer likely removable boring/dead sections or speed-up opportunities.\n"
        "- If prompt gives explicit ranges, prioritize those.\n"
        "- Use speed_video/apply_speed_range when user asks speed-up/faster playback.\n"
        "- For speed suggestions, include speed_multiplier (default 2.0 if unclear).\n"
        "- Use conversation context and existing timeline ranges for iterative follow-ups.\n"
        "- Avoid suggesting duplicate ranges already present unless user asks to modify them.\n"
    )

    safe_summary = (conversation_summary or "").strip()[:MAX_SUMMARY_CHARS]
    compact_history = (chat_history or [])[-MAX_CONTEXT_TURNS:]
    compact_history_lines: list[str] = []
    for turn in compact_history:
        role = str(turn.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(turn.get("content", "")).strip().replace("\n", " ")
        if not content:
            continue
        compact_history_lines.append(f"{role}: {content[:280]}")
    current_state = {
        "trim_ranges": trim_ranges or [],
        "speed_ranges": speed_ranges or [],
    }
    context_block = (
        f"Conversation summary: {safe_summary or '(none)'}\n"
        f"Recent chat turns:\n{chr(10).join(compact_history_lines) if compact_history_lines else '(none)'}\n"
        f"Current timeline state: {json.dumps(current_state)}"
    )

    text_part = f"{instructions}\n{context_block}\nUser prompt: {prompt}"
    parts: list[dict] = [{"text": text_part}]
    if sprite_files:
        parts[0]["text"] += (
            f"\nAttached: {len(sprite_files)} sprite-sheet thumbnail images, sampled evenly "
            f"in chronological order across the full {duration_sec:.1f}s video. Use their visual "
            "content (not just the metadata above) to find real cut points."
        )
        parts.extend(await asyncio.to_thread(_encode_sprite_images, sprite_files))

    strategy = "sprite-vision" if sprite_files else "sprite-summary-prompt"
    parsed = await _call_gemini(api_key, parts)
    raw_suggestions = parsed.get("suggestions", [])
    reasoning = str(parsed.get("reasoning") or "").strip()
    normalized = _normalize_suggestions(raw_suggestions, duration_sec)

    # Bounded self-correction: the model tried (produced suggestions) but every one
    # was invalid — retry exactly once with the validation problem spelled out,
    # rather than silently falling back to the much weaker regex heuristic.
    if not normalized and raw_suggestions:
        retry_parts = list(parts)
        retry_parts.append(
            {
                "text": (
                    "Your previous suggestions were all invalid: every start_sec/end_sec must "
                    f"satisfy 0 <= start_sec < end_sec <= {duration_sec:.3f}. Return corrected JSON "
                    "with the same schema."
                )
            }
        )
        try:
            retry_parsed = await _call_gemini(api_key, retry_parts)
            raw_suggestions = retry_parsed.get("suggestions", [])
            reasoning = str(retry_parsed.get("reasoning") or reasoning).strip()
            normalized = _normalize_suggestions(raw_suggestions, duration_sec)
        except Exception:
            logger.exception("PLAN_EDITS_SELF_CORRECTION_FAILED")

    if not normalized:
        normalized = _fallback_suggest_cuts(prompt, duration_sec)
        if not reasoning:
            reasoning = "Model suggestions were invalid; fell back to regex heuristics."

    if not reasoning:
        reasoning = f"Proposed {len(normalized)} edit(s) based on the prompt and video context."

    result = {
        "plan_id": plan_id,
        "model": GEMINI_MODEL,
        "strategy": strategy,
        "reasoning": reasoning,
        "proposals": [{**item, "id": str(uuid4())} for item in normalized + silence_suggestions],
    }
    logger.info("PLAN_EDITS_NORMALIZED_RESPONSE %s", result)
    return result
