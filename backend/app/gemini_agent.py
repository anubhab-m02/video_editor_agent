from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import httpx

from .validators import parse_time_like

logger = logging.getLogger(__name__)
MAX_CONTEXT_TURNS = 10
MAX_SUMMARY_CHARS = 500


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object in model output.")
    return json.loads(text[start : end + 1])


def _regex_fallback(prompt: str) -> dict:
    lowered = prompt.lower()
    range_match = re.search(r"(?:from|between)\s+([0-9:.]+)\s+(?:to|and|-)\s+([0-9:.]+)", lowered)
    speed_hint = bool(re.search(r"\b(speed\s*up|faster|fast-forward|accelerat(?:e|ed|ing)|\d+(?:\.\d+)?x)\b", lowered))

    if speed_hint and range_match:
        start_raw, end_raw = range_match.groups()
        multiplier_match = re.search(r"(\d+(?:\.\d+)?)\s*x", lowered)
        speed_multiplier = float(multiplier_match.group(1)) if multiplier_match else 2.0
        return {
            "action": "speed_video",
            "operation": "apply_speed_range",
            "start_sec": parse_time_like(start_raw),
            "end_sec": parse_time_like(end_raw),
            "speed_multiplier": speed_multiplier,
            "reason": "Parsed speed range from explicit prompt.",
        }

    if not range_match:
        raise ValueError("Could not parse prompt without Gemini API key.")
    start_raw, end_raw = range_match.groups()
    operation = "extract_range" if "keep only" in lowered else "remove_segment"
    return {
        "action": "trim_video",
        "operation": operation,
        "start_sec": parse_time_like(start_raw),
        "end_sec": parse_time_like(end_raw),
        "reason": "Parsed from explicit range in prompt.",
    }


async def parse_intent(prompt: str, duration_sec: float) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        fallback = _regex_fallback(prompt)
        logger.info("PARSE_INTENT_FALLBACK_RESPONSE %s", fallback)
        return fallback

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    instructions = (
        "Return JSON only with this schema: "
        '{"action":"trim_video|speed_video","operation":"remove_segment|extract_range|apply_speed_range","start_sec":number,"end_sec":number,"speed_multiplier":number,"reason":string}. '
        f"Times must be within 0 and {duration_sec:.3f}. "
        "Interpretation rules: if user says trim/cut/delete/remove from X to Y, set operation=remove_segment. "
        "If user says keep only/highlight/extract from X to Y, set operation=extract_range. "
        "If user says speed up/faster/2x from X to Y, set action=speed_video and operation=apply_speed_range."
    )
    payload = {
        "contents": [{"parts": [{"text": f"{instructions}\nUser prompt: {prompt}"}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    logger.info("GEMINI_RAW_PARSE_INTENT_RESPONSE %s", text)
    parsed = _extract_json(text)
    action = parsed.get("action", "trim_video")
    operation = parsed.get("operation")
    if not operation:
        operation = "apply_speed_range" if action == "speed_video" else "remove_segment"

    speed_multiplier = parsed.get("speed_multiplier", 2.0)
    try:
        speed_multiplier = float(speed_multiplier)
    except Exception:
        speed_multiplier = 2.0

    result = {
        "action": action,
        "operation": operation,
        "start_sec": parse_time_like(parsed["start_sec"]),
        "end_sec": parse_time_like(parsed["end_sec"]),
        "speed_multiplier": speed_multiplier,
        "reason": parsed.get("reason", "Parsed by Gemini."),
    }
    logger.info("PARSE_INTENT_NORMALIZED_RESPONSE %s", result)
    return result


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


async def suggest_cuts_from_sprites(
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
) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    logger.info(
        "SUGGEST_CUTS_REQUEST %s",
        {
            "duration_sec": duration_sec,
            "sprite_interval_sec": sprite_interval_sec,
            "total_frames": total_frames,
            "sheets_count": sheets_count,
            "prompt": prompt,
            "chat_history_count": len(chat_history or []),
            "trim_ranges_count": len(trim_ranges or []),
            "speed_ranges_count": len(speed_ranges or []),
        },
    )
    if not api_key:
        fallback = {
            "model": "fallback",
            "strategy": "rule-based",
            "suggestions": _fallback_suggest_cuts(prompt, duration_sec),
        }
        logger.info("SUGGEST_CUTS_FALLBACK_RESPONSE %s", fallback)
        return fallback

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )

    instructions = (
        "You are an editing planner. Return strict JSON only.\n"
        "Schema: {\"suggestions\":[{\"action\":\"trim_video|speed_video\",\"operation\":\"remove_segment|extract_range|apply_speed_range\",\"start_sec\":number,\"end_sec\":number,\"speed_multiplier\":number,\"reason\":string,\"confidence\":number}]}\n"
        f"Video duration: {duration_sec:.3f}s\n"
        f"Sprite analysis summary: interval={sprite_interval_sec}s, total_frames={total_frames}, sheets={sheets_count}\n"
        "Rules:\n"
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

    payload = {
        "contents": [{"parts": [{"text": f"{instructions}\n{context_block}\nUser prompt: {prompt}"}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    logger.info("GEMINI_RAW_SUGGEST_RESPONSE %s", text)
    parsed = _extract_json(text)
    raw_suggestions = parsed.get("suggestions", [])

    normalized: list[dict] = []
    for item in raw_suggestions:
        try:
            start_sec = parse_time_like(item["start_sec"])
            end_sec = parse_time_like(item["end_sec"])
        except Exception:
            continue
        if not (0 <= start_sec < end_sec <= duration_sec):
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

    if not normalized:
        normalized = _fallback_suggest_cuts(prompt, duration_sec)

    result = {
        "model": "gemini-2.0-flash",
        "strategy": "sprite-summary-prompt",
        "suggestions": normalized,
    }
    logger.info("SUGGEST_CUTS_NORMALIZED_RESPONSE %s", result)
    return result
