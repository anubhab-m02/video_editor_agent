from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional
from uuid import uuid4

import httpx

from .services.media_service import find_persisted_upload
from .validators import parse_time_like, validate_trim
from .video_tools import detect_scene_changes, detect_silence, extract_range, extract_thumbnail

logger = logging.getLogger(__name__)
MAX_CONTEXT_TURNS = 10
MAX_SUMMARY_CHARS = 500
# ponytail: fixed cap keeps the coarse vision pass's cost flat regardless of video
# duration (ADR-0002/ADR-0006) — evenly subsample instead of sending every sheet.
MAX_SPRITE_IMAGES = 6
GEMINI_MODEL = "gemini-3.1-flash-lite"  # ~6x cheaper output than 3.6-flash; matches our small-call-shape workload (few images in, short JSON out), verified live incl. structured output + image input
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
# ADR-0002 escalation cost guards. Verified live (2026-07-25 spike): this model
# handles direct video at ~88 tokens/sec, so a 30s window costs ~2,640 tokens —
# cheap enough that only ONE escalation per plan call is needed, not a budget to
# spend carefully.
ESCALATION_MAX_WINDOW_SEC = 30.0
# Below this duration, skip sprite thumbnails entirely and send the whole video
# directly. At ~88 tok/s (verified), a 120s clip costs only ~10.5K tokens — cheap
# enough that sprite-vision's cost-bounding no longer matters, and real
# motion/audio fixes classification errors (e.g. "downtime" vs "combat") a
# coarse sprite pass can't judge from isolated stills. Escalation is skipped on
# this path — the model already saw everything, there's nothing to zoom into.
DIRECT_VIDEO_MAX_DURATION_SEC = 120.0
# P3-4/L3: on the sprite path (long clips), uniform-interval sampling can land
# entirely between two real cut points — this adds a few extra thumbnails at
# actual detected scene changes, with their exact timestamps called out in the
# prompt, so the coarse pass has some precisely-timestamped signal near real
# transitions instead of only approximate uniform coverage. Bounded and cheap:
# single small frames, not full tiled sheets.
MAX_SCENE_FRAMES = 4
SCENE_CHANGE_THRESHOLD = 0.3
# P3-3: the trigger is user-cue-first, confidence-second. A P3-2 live test showed
# the model's self-reported confidence stays high (0.95) even when WRONG on a
# sampling-gap case — it doesn't "know what it doesn't know" from sparse sprites
# alone. So an explicit ask for precision is the primary signal; the confidence
# threshold is a secondary catch for cases the model itself flags as unsure.
# Both are tunable per-request (dev panel slider, Design Handoff Part 3) with
# this as the fallback default.
ESCALATION_CONFIDENCE_THRESHOLD_DEFAULT = 0.6
_ESCALATION_CUE_KEYWORDS = re.compile(
    r"\b(exact(?:ly)?|precise(?:ly)?|pinpoint|frame[- ]accurate|"
    r"to the (?:second|frame)|the (?:exact|precise) (?:moment|second|point|spot))\b",
    re.IGNORECASE,
)
_SILENCE_KEYWORDS = re.compile(
    r"\b(dead\s*air|silence|silent|pause[s]?|awkward gap[s]?)\b", re.IGNORECASE
)


def _wants_silence_removal(prompt: str) -> bool:
    return bool(_SILENCE_KEYWORDS.search(prompt))


def _wants_precise_escalation(prompt: str) -> bool:
    return bool(_ESCALATION_CUE_KEYWORDS.search(prompt))


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


def _encode_video_clip(path: Path) -> dict:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"inline_data": {"mime_type": "video/mp4", "data": data}}


def perceive(
    *, text: str, image_parts: Optional[list[dict]] = None, video_part: Optional[dict] = None
) -> list[dict]:
    """ADR-0002 hybrid vision seam — the one place perception strategy is chosen.
    Coarse pass: image_parts (sprite thumbnails). Escalation: video_part (a short
    scoped sub-clip, real video). Exactly one of the two is used per call."""
    parts: list[dict] = [{"text": text}]
    if video_part is not None:
        parts.append(video_part)
    elif image_parts:
        parts.extend(image_parts)
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


async def _call_gemini(api_key: str, parts: list[dict]) -> tuple[dict, Optional[int]]:
    """Returns (parsed JSON body, total token count if the API reported one).
    Token count feeds the dev panel's running totals (Design Handoff Part 3) —
    real usage, not the pre-call token_service.py estimate."""
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Key goes in a header, not the URL — httpx exceptions stringify the URL,
        # and that string can end up in an HTTP error response shown to the user.
        response = await client.post(
            GEMINI_URL, json=payload, headers={"x-goog-api-key": api_key}
        )
        response.raise_for_status()
        data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    logger.info("GEMINI_RAW_PLAN_RESPONSE %s", text)
    tokens_used = data.get("usageMetadata", {}).get("totalTokenCount")
    return _extract_json(text), tokens_used


def _fallback_summary(lines: list[str], previous_summary: Optional[str]) -> str:
    user_lines = [line[5:].strip() for line in lines if line.startswith("user:")][-3:]
    if not user_lines:
        return (previous_summary or "").strip()[:MAX_SUMMARY_CHARS]
    return f"Earlier user goals: {' | '.join(user_lines)}"[:MAX_SUMMARY_CHARS]


async def summarize_conversation(
    older_turns: list[dict], previous_summary: Optional[str] = None
) -> dict:
    """P3-5/L1: a real small-model summary of the chat turns that just fell out
    of the raw context window, replacing the old last-3-user-messages heuristic.
    Rolling, not full-history: folds only the NEWLY overflowed turns into
    previous_summary, so cost stays flat regardless of session length — this is
    only called when the raw window actually grows, not on every plan call.
    Text-only (no images/video) — cheap by construction. Falls back to the old
    heuristic with no API key or no real turns to summarize."""
    lines: list[str] = []
    for turn in older_turns:
        role = str(turn.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(turn.get("content", "")).strip().replace("\n", " ")
        if content:
            lines.append(f"{role}: {content[:280]}")

    api_key = os.getenv("GEMINI_API_KEY")
    real_summary = ""
    if api_key and lines:
        text = (
            "Summarize the earlier portion of a video-editing chat in ONE short sentence, "
            "focused on what the user wants done to their video — skip pleasantries and "
            "assistant confirmations.\n"
            + (f"Summary of even-earlier context: {previous_summary}\n" if previous_summary else "")
            + "Earlier turns:\n" + "\n".join(lines) + "\n"
            'Strict JSON only: {"summary": string}'
        )
        try:
            parsed, _tokens_used = await _call_gemini(api_key, perceive(text=text))
            real_summary = str(parsed.get("summary") or "").strip()
        except Exception:
            logger.exception("SUMMARIZE_CONVERSATION_FAILED")

    if real_summary:
        return {"summary": real_summary[:MAX_SUMMARY_CHARS], "model": GEMINI_MODEL}
    return {"summary": _fallback_summary(lines, previous_summary), "model": "fallback"}


async def _escalate(
    *,
    api_key: str,
    candidate: dict,
    duration_sec: float,
    source_path: Path,
    trigger: str,
) -> Optional[tuple[dict, dict]]:
    """Scoped direct-video escalation (ADR-0002): re-perceive a short window around
    a proposal as real video, for a frame-accurate cut the sparse coarse pass
    couldn't locate. Returns (replacement proposal, escalation event for the dev
    panel), or None if the window/extraction/call/parse doesn't produce anything
    usable."""
    center = (candidate["start_sec"] + candidate["end_sec"]) / 2
    window_end = min(duration_sec, center + ESCALATION_MAX_WINDOW_SEC / 2)
    window_start = max(0.0, window_end - ESCALATION_MAX_WINDOW_SEC)
    window_len = window_end - window_start
    if window_len <= 0.5:
        return None

    with tempfile.TemporaryDirectory(prefix="escalation_") as tmp_dir:
        try:
            clip_path = await asyncio.to_thread(
                extract_range, source_path, Path(tmp_dir), window_start, window_end
            )
        except Exception:
            logger.exception("ESCALATION_EXTRACT_FAILED")
            return None

        text = (
            "You are refining an editing suggestion with a closer look. This clip is a "
            f"{window_len:.3f}s window starting at {window_start:.3f}s of the full video, "
            f"so your start_sec/end_sec must be relative to THIS clip (0 to {window_len:.3f}), "
            "not the original video.\n"
            f"Original lower-confidence suggestion in this window: {candidate['reason']}\n"
            "Watch the actual video content and return the precise cut. Strict JSON only: "
            '{"reasoning":string,"suggestions":[{"action":"trim_video|speed_video",'
            '"operation":"remove_segment|extract_range|apply_speed_range","start_sec":number,'
            '"end_sec":number,"speed_multiplier":number,"reason":string,"confidence":number}]}\n'
            f"Each suggestion must satisfy 0 <= start_sec < end_sec <= {window_len:.3f}."
        )
        try:
            video_part = await asyncio.to_thread(_encode_video_clip, clip_path)
            parsed, tokens_used = await _call_gemini(api_key, perceive(text=text, video_part=video_part))
        except Exception:
            logger.exception("ESCALATION_CALL_FAILED")
            return None

    normalized = _normalize_suggestions(parsed.get("suggestions", []), window_len)
    if not normalized:
        return None

    best = max(normalized, key=lambda item: item["confidence"])
    best["start_sec"] = round(best["start_sec"] + window_start, 3)
    best["end_sec"] = round(best["end_sec"] + window_start, 3)
    best["reason"] = f"{best['reason']} (escalated: direct-video zoom-in for precision)"
    best["confidence"] = max(best["confidence"], candidate["confidence"])

    event = {
        "window_start_sec": round(window_start, 3),
        "window_end_sec": round(window_end, 3),
        "trigger": trigger,
        "confidence_before": candidate["confidence"],
        "tokens_used": tokens_used,
    }
    return best, event


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
    escalation_confidence_threshold: Optional[float] = None,
) -> dict:
    """Plan -> validate -> propose (ADR-0003). Below DIRECT_VIDEO_MAX_DURATION_SEC,
    the coarse pass itself uses the whole video directly (real motion/audio, no
    escalation needed); above it, uses sprite thumbnails with scoped escalation.
    One bounded self-correction retry if the model's first attempt yields zero
    valid proposals; falls back to the regex heuristic only if that retry also
    fails. On the sprite path, escalates once (ADR-0002/P3-3): a scoped
    direct-video re-perceive of the least-confident proposal's window, triggered
    by an explicit precision cue in the prompt (primary — the model doesn't
    reliably self-report low confidence on sampling-gap uncertainty) or a
    confidence threshold (secondary catch-all, dev-panel tunable via
    escalation_confidence_threshold). Silence/dead-air detection (X5) is a
    deterministic FFmpeg tool that runs independently of Gemini and merges its
    proposals in regardless of which path produced the rest."""
    confidence_threshold = (
        escalation_confidence_threshold
        if escalation_confidence_threshold is not None
        else ESCALATION_CONFIDENCE_THRESHOLD_DEFAULT
    )
    plan_id = str(uuid4())
    api_key = os.getenv("GEMINI_API_KEY")

    source_path: Optional[Path] = None
    if sprite_job_id and uploads_dir:
        source_path = await asyncio.to_thread(find_persisted_upload, uploads_dir, sprite_job_id)

    use_direct_video = False
    direct_video_part: Optional[dict] = None
    if api_key and source_path is not None and duration_sec <= DIRECT_VIDEO_MAX_DURATION_SEC:
        try:
            direct_video_part = await asyncio.to_thread(_encode_video_clip, source_path)
            use_direct_video = True
        except Exception:
            logger.exception("DIRECT_VIDEO_ENCODE_FAILED")

    sprite_files: list[Path] = []
    if not use_direct_video and api_key and sprite_job_id and sprites_dir:
        sprite_files = await asyncio.to_thread(
            _select_sprite_files, sprites_dir, sprite_job_id, MAX_SPRITE_IMAGES
        )

    silence_suggestions: list[dict] = []
    if _wants_silence_removal(prompt) and source_path is not None:
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
            "used_direct_video": use_direct_video,
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
            "tokens_used": None,
            "escalation": None,
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
    image_parts: Optional[list[dict]] = None
    coarse_video_part: Optional[dict] = None
    if use_direct_video:
        text_part += (
            f"\nAttached: the full {duration_sec:.1f}s video. Watch the actual motion and audio "
            "directly — this is real footage, not thumbnails — to judge what's really happening "
            "(e.g. actual combat vs. downtime), not just what a still frame might suggest."
        )
        coarse_video_part = direct_video_part
        strategy = "direct-video"
    elif sprite_files:
        text_part += (
            f"\nAttached: {len(sprite_files)} sprite-sheet thumbnail images, sampled evenly "
            f"in chronological order across the full {duration_sec:.1f}s video. Use their visual "
            "content (not just the metadata above) to find real cut points."
        )
        image_parts = await asyncio.to_thread(_encode_sprite_images, sprite_files)
        strategy = "sprite-vision"

        # P3-4/L3: a few extra, precisely-timestamped thumbnails at real scene
        # changes, on top of the uniform grid above.
        if source_path is not None:
            try:
                scene_timestamps = await asyncio.to_thread(
                    detect_scene_changes,
                    source_path,
                    threshold=SCENE_CHANGE_THRESHOLD,
                    max_results=MAX_SCENE_FRAMES,
                )
            except Exception:
                logger.exception("SCENE_DETECTION_FAILED")
                scene_timestamps = []
            if scene_timestamps:
                with tempfile.TemporaryDirectory(prefix="scene_frames_") as tmp_dir:
                    try:
                        scene_frame_paths = [
                            await asyncio.to_thread(extract_thumbnail, source_path, Path(tmp_dir), ts)
                            for ts in scene_timestamps
                        ]
                        scene_image_parts = await asyncio.to_thread(_encode_sprite_images, scene_frame_paths)
                    except Exception:
                        logger.exception("SCENE_FRAME_EXTRACT_FAILED")
                        scene_image_parts = []
                if scene_image_parts:
                    image_parts = image_parts + scene_image_parts
                    ts_list = ", ".join(f"{t:.2f}s" for t in scene_timestamps)
                    text_part += (
                        f"\nAlso attached: {len(scene_image_parts)} extra thumbnails at detected "
                        f"scene-change moments, with EXACT timestamps: {ts_list}. Unlike the sprite "
                        "grid above (only approximately evenly spaced), these are precise — use them "
                        "to sharpen cuts near real transitions."
                    )
                    strategy = "sprite-vision+adaptive"
    else:
        strategy = "sprite-summary-prompt"

    parts = perceive(text=text_part, image_parts=image_parts, video_part=coarse_video_part)
    parsed, tokens_used = await _call_gemini(api_key, parts)
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
            retry_parsed, retry_tokens_used = await _call_gemini(api_key, retry_parts)
            raw_suggestions = retry_parsed.get("suggestions", [])
            reasoning = str(retry_parsed.get("reasoning") or reasoning).strip()
            normalized = _normalize_suggestions(raw_suggestions, duration_sec)
            if retry_tokens_used is not None:
                tokens_used = (tokens_used or 0) + retry_tokens_used
        except Exception:
            logger.exception("PLAN_EDITS_SELF_CORRECTION_FAILED")

    # Scoped escalation (ADR-0002/P3-3): trigger is user-cue-first (explicit ask
    # for precision), confidence-threshold second (catches the model's own
    # low-confidence flags). At most one escalation per plan call (cost guard).
    # Skipped when the coarse pass already used direct video (use_direct_video) —
    # the model already saw everything, there's no coarse localization to refine.
    escalation_event: Optional[dict] = None
    if normalized and not use_direct_video and source_path is not None:
        worst_idx, worst = min(enumerate(normalized), key=lambda pair: pair[1]["confidence"])
        wants_precision = _wants_precise_escalation(prompt)
        if wants_precision or worst["confidence"] < confidence_threshold:
            escalated = await _escalate(
                api_key=api_key,
                candidate=worst,
                duration_sec=duration_sec,
                source_path=source_path,
                trigger="user_cue" if wants_precision else "low_confidence",
            )
            if escalated is not None:
                replacement, escalation_event = escalated
                normalized[worst_idx] = replacement
                strategy = f"{strategy}+escalation"
                if escalation_event["tokens_used"] is not None:
                    tokens_used = (tokens_used or 0) + escalation_event["tokens_used"]

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
        "tokens_used": tokens_used,
        "escalation": escalation_event,
    }
    logger.info("PLAN_EDITS_NORMALIZED_RESPONSE %s", result)
    return result
