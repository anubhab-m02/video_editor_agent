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


def _gemini_json_response(suggestions):
    import json

    text = json.dumps({"suggestions": suggestions})
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_suggest_cuts_attaches_sprite_images_when_available(tmp_path, monkeypatch):
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
        gemini_agent.suggest_cuts_from_sprites(
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


def test_suggest_cuts_falls_back_to_text_only_without_sprite_job_id(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    captured = {}

    async def fake_post(self, url, json=None):
        captured["payload"] = json
        return _FakeResponse(_gemini_json_response([]))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(
        gemini_agent.suggest_cuts_from_sprites(
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
