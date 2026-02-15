from pathlib import Path
import sys
import asyncio

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app
from app.gemini_agent import parse_intent


client = TestClient(app)


def test_token_estimate_endpoint_returns_comparison():
    response = client.post(
        "/analyze/token-estimate",
        json={
            "duration_sec": 120,
            "interval_sec": 1.0,
            "columns": 8,
            "rows": 8,
            "thumb_width": 256,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["duration_sec"] == 120
    assert data["direct_video_tokens_est"] > 0
    assert data["sprite_tokens_est"] > 0
    assert "recommendation" in data


def test_token_estimate_rejects_invalid_duration():
    response = client.post(
        "/analyze/token-estimate",
        json={
            "duration_sec": 0,
            "interval_sec": 1.0,
            "columns": 8,
            "rows": 8,
            "thumb_width": 256,
        },
    )
    assert response.status_code == 422


def test_suggest_cuts_fallback_explicit_range(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post(
        "/ai/suggest-cuts-from-sprites",
        json={
            "prompt": "Cut from 4 to 5 seconds",
            "duration_sec": 20,
            "sprite_interval_sec": 1.0,
            "total_frames": 21,
            "sheets_count": 1,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "fallback"
    assert data["strategy"] == "rule-based"
    assert len(data["suggestions"]) >= 1
    first = data["suggestions"][0]
    assert first["action"] == "trim_video"
    assert first["operation"] == "remove_segment"
    assert first["start_sec"] == 4
    assert first["end_sec"] == 5


def test_suggest_cuts_fallback_speed_range(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post(
        "/ai/suggest-cuts-from-sprites",
        json={
            "prompt": "Speed up 2x from 4 to 5 seconds",
            "duration_sec": 20,
            "sprite_interval_sec": 1.0,
            "total_frames": 21,
            "sheets_count": 1,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["suggestions"]) >= 1
    first = data["suggestions"][0]
    assert first["action"] == "speed_video"
    assert first["operation"] == "apply_speed_range"
    assert first["start_sec"] == 4
    assert first["end_sec"] == 5
    assert first["speed_multiplier"] == 2


def test_suggest_cuts_accepts_context_fields(monkeypatch):
    import app.main as main

    captured = {}

    async def fake_suggest_cuts_from_sprites(**kwargs):
        captured.update(kwargs)
        return {
            "model": "fake",
            "strategy": "test",
            "suggestions": [],
        }

    monkeypatch.setattr(main, "suggest_cuts_from_sprites", fake_suggest_cuts_from_sprites)

    response = client.post(
        "/ai/suggest-cuts-from-sprites",
        json={
            "prompt": "Make intro faster",
            "duration_sec": 8,
            "sprite_interval_sec": 0.25,
            "total_frames": 32,
            "sheets_count": 1,
            "chat_history": [
                {"role": "user", "content": "trim intro"},
                {"role": "assistant", "content": "applied"},
            ],
            "conversation_summary": "Earlier user goals: trim pauses",
            "trim_ranges": [{"start": 1.0, "end": 1.5}],
            "speed_ranges": [{"start": 2.0, "end": 3.0, "speed": 2.0}],
        },
    )
    assert response.status_code == 200
    assert captured["chat_history"][0]["role"] == "user"
    assert captured["conversation_summary"] == "Earlier user goals: trim pauses"
    assert captured["trim_ranges"] == [{"start": 1.0, "end": 1.5}]
    assert captured["speed_ranges"] == [{"start": 2.0, "end": 3.0, "speed": 2.0}]


def test_parse_intent_fallback_speed_range(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    intent = asyncio.run(parse_intent("Speed up 2x from 4 to 5 seconds", 10))
    assert intent["action"] == "speed_video"
    assert intent["operation"] == "apply_speed_range"
    assert intent["start_sec"] == 4
    assert intent["end_sec"] == 5
    assert intent["speed_multiplier"] == 2


def test_token_estimate_from_file_rejects_over_max_duration(monkeypatch, tmp_path):
    import app.main as main

    monkeypatch.setattr(main, "MAX_VIDEO_DURATION_SEC", 10.0)

    fake_file = tmp_path / "sample.mp4"
    fake_file.write_bytes(b"dummy")

    async def fake_save_upload_file(*, file, upload_dir, max_file_size_mb=None):
        return fake_file

    monkeypatch.setattr(main, "save_upload_file", fake_save_upload_file)
    monkeypatch.setattr(main, "probe_duration_or_cleanup", lambda _: 12.0)

    response = client.post(
        "/analyze/token-estimate-from-file",
        files={"file": ("sample.mp4", b"dummy", "video/mp4")},
    )
    assert response.status_code == 400
    assert "exceeds maximum" in response.json()["detail"]


def test_export_from_file_rejects_invalid_speed(monkeypatch, tmp_path):
    import app.main as main

    source_file = tmp_path / "source.mp4"
    source_file.write_bytes(b"dummy")

    async def fake_save_upload_file(*, file, upload_dir, max_file_size_mb=None):
        return source_file

    monkeypatch.setattr(main, "save_upload_file", fake_save_upload_file)
    monkeypatch.setattr(main, "probe_duration_or_cleanup", lambda _: 8.0)

    response = client.post(
        "/export/from-file",
        data={"trim_ranges": "[]", "speed": "3x"},
        files={"file": ("sample.mp4", b"dummy", "video/mp4")},
    )
    assert response.status_code == 400
    assert "Only 1x and 2x are supported" in response.json()["detail"]


def test_export_from_file_applies_segment_speed_ranges(monkeypatch, tmp_path):
    import app.main as main

    source_file = tmp_path / "source.mp4"
    source_file.write_bytes(b"dummy")
    rendered_file = tmp_path / "rendered.mp4"
    rendered_file.write_bytes(b"rendered")

    async def fake_save_upload_file(*, file, upload_dir, max_file_size_mb=None):
        return source_file

    monkeypatch.setattr(main, "save_upload_file", fake_save_upload_file)
    monkeypatch.setattr(main, "probe_duration_or_cleanup", lambda _: 8.0)
    monkeypatch.setattr(main, "get_duration_sec", lambda _: 6.0)

    captured = {}

    def fake_render_segments_with_speed(*, input_path, output_dir, segments):
        captured["segments"] = segments
        return rendered_file

    monkeypatch.setattr(main, "render_segments_with_speed", fake_render_segments_with_speed)
    monkeypatch.setattr(main, "remove_segments_and_stitch", lambda **kwargs: rendered_file)
    monkeypatch.setattr(main, "extract_range", lambda **kwargs: rendered_file)

    response = client.post(
        "/export/from-file",
        data={
            "trim_ranges": "[]",
            "speed_ranges": '[{"start":1.0,"end":3.0,"speed":2}]',
        },
        files={"file": ("sample.mp4", b"dummy", "video/mp4")},
    )
    assert response.status_code == 200
    assert captured["segments"] == [(0.0, 1.0, 1.0), (1.0, 3.0, 2.0), (3.0, 8.0, 1.0)]


def test_export_splits_full_speed_when_trims_are_present(monkeypatch, tmp_path):
    import app.main as main

    source_file = tmp_path / "source.mp4"
    source_file.write_bytes(b"dummy")
    rendered_file = tmp_path / "rendered.mp4"
    rendered_file.write_bytes(b"rendered")

    async def fake_save_upload_file(*, file, upload_dir, max_file_size_mb=None):
        return source_file

    monkeypatch.setattr(main, "save_upload_file", fake_save_upload_file)
    monkeypatch.setattr(main, "probe_duration_or_cleanup", lambda _: 8.0)
    monkeypatch.setattr(main, "get_duration_sec", lambda _: 5.0)

    captured = {}

    def fake_render_segments_with_speed(*, input_path, output_dir, segments):
        captured["segments"] = segments
        return rendered_file

    monkeypatch.setattr(main, "render_segments_with_speed", fake_render_segments_with_speed)
    monkeypatch.setattr(main, "remove_segments_and_stitch", lambda **kwargs: rendered_file)
    monkeypatch.setattr(main, "extract_range", lambda **kwargs: rendered_file)

    response = client.post(
        "/export/from-file",
        data={
            "trim_ranges": '[{"start":3.0,"end":4.0},{"start":6.0,"end":7.0}]',
            "speed_ranges": '[{"start":0.0,"end":8.0,"speed":2}]',
        },
        files={"file": ("sample.mp4", b"dummy", "video/mp4")},
    )
    assert response.status_code == 200
    assert captured["segments"] == [(0.0, 3.0, 2.0), (4.0, 6.0, 2.0), (7.0, 8.0, 2.0)]


def test_edit_request_applies_speed_range(monkeypatch, tmp_path):
    import app.main as main

    source_file = tmp_path / "source.mp4"
    source_file.write_bytes(b"dummy")
    rendered_file = tmp_path / "rendered.mp4"
    rendered_file.write_bytes(b"rendered")

    main.video_sessions["speed-test"] = {
        "input_path": source_file,
        "duration_sec": 8.0,
        "filename": "source.mp4",
    }

    async def fake_parse_intent(prompt: str, duration_sec: float):
        return {
            "action": "speed_video",
            "operation": "apply_speed_range",
            "start_sec": 1.0,
            "end_sec": 3.0,
            "speed_multiplier": 2.0,
            "reason": "Speed up highlighted section.",
        }

    captured = {}

    def fake_render_segments_with_speed(*, input_path, output_dir, segments):
        captured["segments"] = segments
        return rendered_file

    monkeypatch.setattr(main, "parse_intent", fake_parse_intent)
    monkeypatch.setattr(main, "render_segments_with_speed", fake_render_segments_with_speed)

    response = client.post(
        "/edit-request",
        json={
            "video_id": "speed-test",
            "prompt": "Speed this section",
        },
    )

    main.video_sessions.pop("speed-test", None)

    assert response.status_code == 200
    data = response.json()
    assert data["action"] == "speed_video"
    assert data["operation"] == "apply_speed_range"
    assert captured["segments"] == [(0.0, 1.0, 1.0), (1.0, 3.0, 2.0), (3.0, 8.0, 1.0)]
