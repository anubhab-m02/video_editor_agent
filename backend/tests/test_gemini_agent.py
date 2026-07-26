import asyncio
from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import httpx

from app import gemini_agent


def _make_sheets(tmp_path: Path, job_id: str, count: int) -> Path:
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    for i in range(1, count + 1):
        (job_dir / f"sheet_{i:03d}.png").write_bytes(b"fake-png-bytes")
    return tmp_path


def test_select_sprite_files_returns_all_when_under_limit(tmp_path):
    sprites_dir = _make_sheets(tmp_path, "job1", 3)
    files = gemini_agent._select_sprite_files(sprites_dir, "job1", limit=6)
    assert len(files) == 3
    assert files == sorted(files)


def test_select_sprite_files_subsamples_evenly_when_over_limit(tmp_path):
    sprites_dir = _make_sheets(tmp_path, "job2", 30)
    files = gemini_agent._select_sprite_files(sprites_dir, "job2", limit=6)
    assert len(files) == 6
    assert files == sorted(files)


def test_select_sprite_files_missing_job_returns_empty(tmp_path):
    files = gemini_agent._select_sprite_files(tmp_path, "does-not-exist", limit=6)
    assert files == []


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _gemini_json_response(suggestions, reasoning="test reasoning"):
    import json

    text = json.dumps({"reasoning": reasoning, "suggestions": suggestions})
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_plan_edits_attaches_sprite_images_when_available(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    sprites_dir = _make_sheets(tmp_path, "job3", 10)

    captured = {}

    async def fake_post(self, url, json=None, headers=None):
        captured["payload"] = json
        return _FakeResponse(
            _gemini_json_response(
                [
                    {
                        "action": "trim_video",
                        "operation": "remove_segment",
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                        "reason": "boring",
                        "confidence": 0.8,
                    }
                ]
            )
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(
        gemini_agent.plan_edits(
            prompt="find the boring part",
            duration_sec=20.0,
            sprite_interval_sec=1.0,
            total_frames=20,
            sheets_count=1,
            sprite_job_id="job3",
            sprites_dir=sprites_dir,
        )
    )

    parts = captured["payload"]["contents"][0]["parts"]
    image_parts = [p for p in parts if "inline_data" in p]
    assert len(image_parts) == 6
    assert result["strategy"] == "sprite-vision"
    assert result["plan_id"]
    assert result["reasoning"] == "test reasoning"
    assert result["proposals"][0]["id"]


def test_plan_edits_falls_back_to_text_only_without_sprite_job_id(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    captured = {}

    async def fake_post(self, url, json=None, headers=None):
        captured["payload"] = json
        return _FakeResponse(_gemini_json_response([]))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(
        gemini_agent.plan_edits(
            prompt="find the boring part",
            duration_sec=20.0,
            sprite_interval_sec=1.0,
            total_frames=20,
            sheets_count=1,
        )
    )

    parts = captured["payload"]["contents"][0]["parts"]
    image_parts = [p for p in parts if "inline_data" in p]
    assert len(image_parts) == 0
    assert result["strategy"] == "sprite-summary-prompt"


def test_plan_edits_self_corrects_after_invalid_first_attempt(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    call_count = {"n": 0}

    async def fake_post(self, url, json=None, headers=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Out of range (end_sec > duration): every item should fail validation.
            return _FakeResponse(
                _gemini_json_response(
                    [{"action": "trim_video", "start_sec": 1.0, "end_sec": 999.0, "reason": "bad", "confidence": 0.5}]
                )
            )
        return _FakeResponse(
            _gemini_json_response(
                [{"action": "trim_video", "start_sec": 1.0, "end_sec": 2.0, "reason": "corrected", "confidence": 0.7}],
                reasoning="corrected reasoning",
            )
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(
        gemini_agent.plan_edits(
            prompt="find the boring part",
            duration_sec=20.0,
            sprite_interval_sec=1.0,
            total_frames=20,
            sheets_count=1,
        )
    )

    assert call_count["n"] == 2
    assert len(result["proposals"]) == 1
    assert result["proposals"][0]["reason"] == "corrected"
    assert result["reasoning"] == "corrected reasoning"


def test_plan_edits_falls_back_to_regex_if_retry_also_invalid(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    async def fake_post(self, url, json=None, headers=None):
        return _FakeResponse(
            _gemini_json_response(
                [{"action": "trim_video", "start_sec": 1.0, "end_sec": 999.0, "reason": "bad", "confidence": 0.5}]
            )
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(
        gemini_agent.plan_edits(
            prompt="Cut from 4 to 5 seconds",
            duration_sec=20.0,
            sprite_interval_sec=1.0,
            total_frames=20,
            sheets_count=1,
        )
    )

    assert len(result["proposals"]) >= 1
    assert result["proposals"][0]["start_sec"] == 4
    assert result["proposals"][0]["end_sec"] == 5


def test_plan_edits_adds_silence_proposals_when_requested(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    (uploads_dir / "job5.mp4").write_bytes(b"fake-video-bytes")

    monkeypatch.setattr(gemini_agent, "detect_silence", lambda path: [(2.0, 4.5)])

    result = asyncio.run(
        gemini_agent.plan_edits(
            prompt="remove the dead air",
            duration_sec=20.0,
            sprite_interval_sec=1.0,
            total_frames=20,
            sheets_count=1,
            sprite_job_id="job5",
            uploads_dir=uploads_dir,
        )
    )

    silence_items = [p for p in result["proposals"] if "silence" in p["reason"].lower()]
    assert len(silence_items) == 1
    assert silence_items[0]["start_sec"] == 2.0
    assert silence_items[0]["end_sec"] == 4.5
    assert silence_items[0]["confidence"] == 0.9


def test_plan_edits_skips_silence_detection_without_keyword(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    (uploads_dir / "job6.mp4").write_bytes(b"fake-video-bytes")

    called = {"n": 0}

    def fake_detect_silence(path):
        called["n"] += 1
        return [(2.0, 4.5)]

    monkeypatch.setattr(gemini_agent, "detect_silence", fake_detect_silence)

    result = asyncio.run(
        gemini_agent.plan_edits(
            prompt="make it faster",
            duration_sec=20.0,
            sprite_interval_sec=1.0,
            total_frames=20,
            sheets_count=1,
            sprite_job_id="job6",
            uploads_dir=uploads_dir,
        )
    )

    assert called["n"] == 0
    assert not any("silence" in p["reason"].lower() for p in result["proposals"])


def test_plan_edits_skips_silence_detection_when_upload_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()

    called = {"n": 0}

    def fake_detect_silence(path):
        called["n"] += 1
        return [(2.0, 4.5)]

    monkeypatch.setattr(gemini_agent, "detect_silence", fake_detect_silence)

    result = asyncio.run(
        gemini_agent.plan_edits(
            prompt="remove the dead air",
            duration_sec=20.0,
            sprite_interval_sec=1.0,
            total_frames=20,
            sheets_count=1,
            sprite_job_id="missing-job",
            uploads_dir=uploads_dir,
        )
    )

    assert called["n"] == 0
    assert not any("silence" in p["reason"].lower() for p in result["proposals"])


def test_plan_edits_escalates_low_confidence_proposal(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    sprites_dir = _make_sheets(tmp_path, "job8", 3)
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    (uploads_dir / "job8.mp4").write_bytes(b"fake-video-bytes")

    # Avoid real ffmpeg/decoding: escalation only needs a Path back, and only needs
    # something base64-encodable for the (mocked) Gemini call.
    monkeypatch.setattr(
        gemini_agent, "extract_range", lambda input_path, output_dir, start_sec, end_sec: input_path
    )
    monkeypatch.setattr(
        gemini_agent, "_encode_video_clip", lambda path: {"inline_data": {"mime_type": "video/mp4", "data": "FAKE"}}
    )

    call_count = {"n": 0}

    async def fake_post(self, url, json=None, headers=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Coarse pass: one low-confidence proposal, well below threshold.
            return _FakeResponse(
                _gemini_json_response(
                    [
                        {
                            "action": "trim_video",
                            "start_sec": 30.0,
                            "end_sec": 34.0,
                            "reason": "maybe boring",
                            "confidence": 0.4,
                        }
                    ]
                )
            )
        # Escalation call: times are relative to the scoped window, not the full video.
        return _FakeResponse(
            _gemini_json_response(
                [
                    {
                        "action": "trim_video",
                        "start_sec": 5.0,
                        "end_sec": 9.0,
                        "reason": "precise cut",
                        "confidence": 0.95,
                    }
                ],
                reasoning="escalated reasoning",
            )
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(
        gemini_agent.plan_edits(
            prompt="find the exact boring moment",
            duration_sec=180.0,  # above DIRECT_VIDEO_MAX_DURATION_SEC to exercise the sprite+escalation path
            sprite_interval_sec=1.0,
            total_frames=60,
            sheets_count=1,
            sprite_job_id="job8",
            sprites_dir=sprites_dir,
            uploads_dir=uploads_dir,
        )
    )

    assert call_count["n"] == 2
    assert result["strategy"] == "sprite-vision+escalation"
    assert len(result["proposals"]) == 1
    proposal = result["proposals"][0]
    # window = [17, 47] (center 32 +/- 15, clamped); local [5,9] maps back to [22,26].
    assert proposal["start_sec"] == 22.0
    assert proposal["end_sec"] == 26.0
    assert "escalated" in proposal["reason"].lower()
    assert proposal["confidence"] == 0.95
    # Prompt contains "exact" -> user_cue fires (also independently true here since
    # confidence 0.4 is below the default threshold too, but user_cue takes priority).
    assert result["escalation"]["trigger"] == "user_cue"
    assert result["escalation"]["window_start_sec"] == 17.0
    assert result["escalation"]["window_end_sec"] == 47.0
    assert result["escalation"]["confidence_before"] == 0.4


def test_plan_edits_escalates_on_confidence_alone_without_cue(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    sprites_dir = _make_sheets(tmp_path, "job10", 3)
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    (uploads_dir / "job10.mp4").write_bytes(b"fake-video-bytes")

    monkeypatch.setattr(
        gemini_agent, "extract_range", lambda input_path, output_dir, start_sec, end_sec: input_path
    )
    monkeypatch.setattr(
        gemini_agent, "_encode_video_clip", lambda path: {"inline_data": {"mime_type": "video/mp4", "data": "FAKE"}}
    )

    call_count = {"n": 0}

    async def fake_post(self, url, json=None, headers=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeResponse(
                _gemini_json_response(
                    [{"action": "trim_video", "start_sec": 30.0, "end_sec": 34.0, "reason": "unsure", "confidence": 0.3}]
                )
            )
        return _FakeResponse(
            _gemini_json_response(
                [{"action": "trim_video", "start_sec": 5.0, "end_sec": 9.0, "reason": "refined", "confidence": 0.9}]
            )
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    # No precision cue words in this prompt — only the confidence threshold can trigger.
    result = asyncio.run(
        gemini_agent.plan_edits(
            prompt="cut the boring part",
            duration_sec=180.0,  # above DIRECT_VIDEO_MAX_DURATION_SEC to exercise the sprite+escalation path
            sprite_interval_sec=1.0,
            total_frames=60,
            sheets_count=1,
            sprite_job_id="job10",
            sprites_dir=sprites_dir,
            uploads_dir=uploads_dir,
        )
    )

    assert call_count["n"] == 2
    assert result["strategy"] == "sprite-vision+escalation"
    assert result["escalation"]["trigger"] == "low_confidence"


def test_plan_edits_escalation_threshold_override_forces_escalation(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    sprites_dir = _make_sheets(tmp_path, "job11", 3)
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    (uploads_dir / "job11.mp4").write_bytes(b"fake-video-bytes")

    monkeypatch.setattr(
        gemini_agent, "extract_range", lambda input_path, output_dir, start_sec, end_sec: input_path
    )
    monkeypatch.setattr(
        gemini_agent, "_encode_video_clip", lambda path: {"inline_data": {"mime_type": "video/mp4", "data": "FAKE"}}
    )

    call_count = {"n": 0}

    async def fake_post(self, url, json=None, headers=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Confidence 0.8 clears the default threshold (0.6) and has no cue word —
            # would NOT escalate without the dev-panel override.
            return _FakeResponse(
                _gemini_json_response(
                    [{"action": "trim_video", "start_sec": 4.0, "end_sec": 8.0, "reason": "fairly sure", "confidence": 0.8}]
                )
            )
        return _FakeResponse(
            _gemini_json_response(
                [{"action": "trim_video", "start_sec": 1.0, "end_sec": 2.0, "reason": "refined", "confidence": 0.9}]
            )
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(
        gemini_agent.plan_edits(
            prompt="cut the boring part",
            duration_sec=180.0,  # above DIRECT_VIDEO_MAX_DURATION_SEC to exercise the sprite+escalation path
            sprite_interval_sec=1.0,
            total_frames=60,
            sheets_count=1,
            sprite_job_id="job11",
            sprites_dir=sprites_dir,
            uploads_dir=uploads_dir,
            escalation_confidence_threshold=0.95,
        )
    )

    assert call_count["n"] == 2
    assert result["strategy"] == "sprite-vision+escalation"
    assert result["escalation"]["trigger"] == "low_confidence"


def test_plan_edits_does_not_escalate_high_confidence_proposal(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    sprites_dir = _make_sheets(tmp_path, "job9", 3)
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    (uploads_dir / "job9.mp4").write_bytes(b"fake-video-bytes")

    called = {"n": 0}
    monkeypatch.setattr(
        gemini_agent,
        "extract_range",
        lambda *a, **kw: called.__setitem__("n", called["n"] + 1) or a[0],
    )

    async def fake_post(self, url, json=None, headers=None):
        return _FakeResponse(
            _gemini_json_response(
                [
                    {
                        "action": "trim_video",
                        "start_sec": 4.0,
                        "end_sec": 8.0,
                        "reason": "confident cut",
                        "confidence": 0.9,
                    }
                ]
            )
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(
        gemini_agent.plan_edits(
            prompt="cut the boring part",
            duration_sec=180.0,  # above DIRECT_VIDEO_MAX_DURATION_SEC to exercise the sprite+escalation path
            sprite_interval_sec=1.0,
            total_frames=60,
            sheets_count=1,
            sprite_job_id="job9",
            sprites_dir=sprites_dir,
            uploads_dir=uploads_dir,
        )
    )

    assert called["n"] == 0
    assert result["strategy"] == "sprite-vision"
    assert result["proposals"][0]["confidence"] == 0.9


def test_plan_edits_uses_direct_video_for_short_clip(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    (uploads_dir / "job12.mp4").write_bytes(b"fake-video-bytes")

    escalate_called = {"n": 0}
    monkeypatch.setattr(
        gemini_agent,
        "_escalate",
        lambda **kw: escalate_called.__setitem__("n", escalate_called["n"] + 1),
    )

    captured = {}

    async def fake_post(self, url, json=None, headers=None):
        captured["payload"] = json
        return _FakeResponse(
            _gemini_json_response(
                [{"action": "trim_video", "start_sec": 4.0, "end_sec": 8.0, "reason": "actual combat", "confidence": 0.6}]
            )
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(
        gemini_agent.plan_edits(
            prompt="find the kills",
            duration_sec=46.5,  # below DIRECT_VIDEO_MAX_DURATION_SEC
            sprite_interval_sec=1.0,
            total_frames=46,
            sheets_count=1,
            sprite_job_id="job12",
            uploads_dir=uploads_dir,
        )
    )

    parts = captured["payload"]["contents"][0]["parts"]
    video_parts = [p for p in parts if "inline_data" in p and p["inline_data"]["mime_type"] == "video/mp4"]
    image_parts = [p for p in parts if "inline_data" in p and p["inline_data"]["mime_type"] == "image/png"]
    assert len(video_parts) == 1
    assert len(image_parts) == 0
    assert result["strategy"] == "direct-video"
    # Escalation makes no sense here — the model already saw the whole clip directly.
    assert escalate_called["n"] == 0


def test_plan_edits_uses_sprites_for_long_clip(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    sprites_dir = _make_sheets(tmp_path, "job13", 3)
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    (uploads_dir / "job13.mp4").write_bytes(b"fake-video-bytes")

    captured = {}

    async def fake_post(self, url, json=None, headers=None):
        captured["payload"] = json
        return _FakeResponse(
            _gemini_json_response(
                [{"action": "trim_video", "start_sec": 4.0, "end_sec": 8.0, "reason": "boring", "confidence": 0.9}]
            )
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(
        gemini_agent.plan_edits(
            prompt="find the kills",
            duration_sec=180.0,  # above DIRECT_VIDEO_MAX_DURATION_SEC
            sprite_interval_sec=1.0,
            total_frames=180,
            sheets_count=1,
            sprite_job_id="job13",
            sprites_dir=sprites_dir,
            uploads_dir=uploads_dir,
        )
    )

    parts = captured["payload"]["contents"][0]["parts"]
    video_parts = [p for p in parts if "inline_data" in p and p["inline_data"]["mime_type"] == "video/mp4"]
    image_parts = [p for p in parts if "inline_data" in p and p["inline_data"]["mime_type"] == "image/png"]
    assert len(video_parts) == 0
    assert len(image_parts) == 3
    assert result["strategy"] == "sprite-vision"


def test_plan_edits_adds_scene_change_frames_when_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    sprites_dir = _make_sheets(tmp_path, "job14", 3)
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    (uploads_dir / "job14.mp4").write_bytes(b"fake-video-bytes")

    monkeypatch.setattr(
        gemini_agent, "detect_scene_changes", lambda source_path, threshold, max_results: [12.5, 47.25]
    )
    monkeypatch.setattr(
        gemini_agent, "extract_thumbnail", lambda source_path, output_dir, timestamp_sec: source_path
    )

    captured = {}

    async def fake_post(self, url, json=None, headers=None):
        captured["payload"] = json
        return _FakeResponse(
            _gemini_json_response(
                [{"action": "trim_video", "start_sec": 4.0, "end_sec": 8.0, "reason": "boring", "confidence": 0.9}]
            )
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(
        gemini_agent.plan_edits(
            prompt="find the boring part",
            duration_sec=180.0,
            sprite_interval_sec=1.0,
            total_frames=180,
            sheets_count=1,
            sprite_job_id="job14",
            sprites_dir=sprites_dir,
            uploads_dir=uploads_dir,
        )
    )

    parts = captured["payload"]["contents"][0]["parts"]
    image_parts = [p for p in parts if "inline_data" in p and p["inline_data"]["mime_type"] == "image/png"]
    # 3 uniform sprite sheets + 2 scene-change frames.
    assert len(image_parts) == 5
    assert result["strategy"] == "sprite-vision+adaptive"
    assert "12.50s" in captured["payload"]["contents"][0]["parts"][0]["text"]
    assert "47.25s" in captured["payload"]["contents"][0]["parts"][0]["text"]


def test_plan_edits_strategy_unchanged_when_no_scene_changes_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    sprites_dir = _make_sheets(tmp_path, "job15", 3)
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    (uploads_dir / "job15.mp4").write_bytes(b"fake-video-bytes")

    monkeypatch.setattr(gemini_agent, "detect_scene_changes", lambda source_path, threshold, max_results: [])

    async def fake_post(self, url, json=None, headers=None):
        return _FakeResponse(
            _gemini_json_response(
                [{"action": "trim_video", "start_sec": 4.0, "end_sec": 8.0, "reason": "boring", "confidence": 0.9}]
            )
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(
        gemini_agent.plan_edits(
            prompt="find the boring part",
            duration_sec=180.0,
            sprite_interval_sec=1.0,
            total_frames=180,
            sheets_count=1,
            sprite_job_id="job15",
            sprites_dir=sprites_dir,
            uploads_dir=uploads_dir,
        )
    )

    assert result["strategy"] == "sprite-vision"


def test_summarize_conversation_uses_real_model_call(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    async def fake_post(self, url, json=None, headers=None):
        return _FakeResponse(
            {
                "candidates": [
                    {"content": {"parts": [{"text": '{"summary": "User wants dead air removed from the intro."}'}]}}
                ]
            }
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(
        gemini_agent.summarize_conversation(
            older_turns=[
                {"role": "user", "content": "remove the dead air at the start"},
                {"role": "assistant", "content": "Removed 3.2s of silence."},
            ],
            previous_summary=None,
        )
    )

    assert result["summary"] == "User wants dead air removed from the intro."
    assert result["model"] == "gemini-3.1-flash-lite"


def test_summarize_conversation_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = asyncio.run(
        gemini_agent.summarize_conversation(
            older_turns=[
                {"role": "user", "content": "cut the intro"},
                {"role": "assistant", "content": "Done."},
                {"role": "user", "content": "also speed up the outro"},
            ],
            previous_summary=None,
        )
    )

    assert result["model"] == "fallback"
    assert "cut the intro" in result["summary"]
    assert "speed up the outro" in result["summary"]


def test_summarize_conversation_falls_back_when_model_call_fails(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    async def fake_post(self, url, json=None, headers=None):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(
        gemini_agent.summarize_conversation(
            older_turns=[{"role": "user", "content": "trim the boring bits"}],
            previous_summary="Earlier user goals: cut the intro",
        )
    )

    assert result["model"] == "fallback"
    assert "trim the boring bits" in result["summary"]


def test_summarize_conversation_falls_back_to_previous_summary_when_no_turns(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = asyncio.run(
        gemini_agent.summarize_conversation(older_turns=[], previous_summary="Earlier user goals: cut the intro")
    )

    assert result["summary"] == "Earlier user goals: cut the intro"
    assert result["model"] == "fallback"
