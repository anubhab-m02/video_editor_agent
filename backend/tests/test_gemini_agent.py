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

    async def fake_post(self, url, json=None):
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

    async def fake_post(self, url, json=None):
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

    async def fake_post(self, url, json=None):
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

    async def fake_post(self, url, json=None):
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
