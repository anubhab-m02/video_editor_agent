from pathlib import Path
import os
import sys
import time

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app


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


def test_agent_plan_fallback_explicit_range(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post(
        "/agent/plan",
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
    assert data["plan_id"]
    assert data["reasoning"]
    assert len(data["proposals"]) >= 1
    first = data["proposals"][0]
    assert first["id"]
    assert first["action"] == "trim_video"
    assert first["operation"] == "remove_segment"
    assert first["start_sec"] == 4
    assert first["end_sec"] == 5


def test_agent_plan_fallback_speed_range(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post(
        "/agent/plan",
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
    assert len(data["proposals"]) >= 1
    first = data["proposals"][0]
    assert first["action"] == "speed_video"
    assert first["operation"] == "apply_speed_range"
    assert first["start_sec"] == 4
    assert first["end_sec"] == 5
    assert first["speed_multiplier"] == 2


def test_agent_plan_accepts_context_fields(monkeypatch):
    import app.main as main

    captured = {}

    async def fake_plan_edits(**kwargs):
        captured.update(kwargs)
        return {
            "plan_id": "fake-plan",
            "reasoning": "test",
            "model": "fake",
            "strategy": "test",
            "proposals": [],
        }

    monkeypatch.setattr(main, "plan_edits", fake_plan_edits)

    response = client.post(
        "/agent/plan",
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


def test_api_key_required_when_set(monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "API_KEY", "secret123")

    no_header = client.post("/analyze/token-estimate", json={"duration_sec": 10})
    assert no_header.status_code == 401

    wrong_header = client.post(
        "/analyze/token-estimate",
        json={"duration_sec": 10},
        headers={"X-API-Key": "wrong"},
    )
    assert wrong_header.status_code == 401

    right_header = client.post(
        "/analyze/token-estimate",
        json={"duration_sec": 10},
        headers={"X-API-Key": "secret123"},
    )
    assert right_header.status_code == 200

    health = client.get("/health")
    assert health.status_code == 200


def test_api_key_not_required_when_unset(monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "API_KEY", "")

    response = client.post("/analyze/token-estimate", json={"duration_sec": 10})
    assert response.status_code == 200


def test_sweep_stale_media_deletes_stale_files_only(tmp_path, monkeypatch):
    import app.main as main

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(main, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(main, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(main, "OUTPUT_TTL_MIN", 60.0)

    old_output = output_dir / "old.mp4"
    old_output.write_bytes(b"x")
    old_upload = upload_dir / "old-source.mp4"
    old_upload.write_bytes(b"x")
    old_time = time.time() - 3600
    os.utime(old_output, (old_time, old_time))
    os.utime(old_upload, (old_time, old_time))

    new_output = output_dir / "new.mp4"
    new_output.write_bytes(b"x")

    main._sweep_stale_media()

    assert not old_output.exists()
    assert not old_upload.exists()
    assert new_output.exists()


def test_sweep_stale_media_disabled_when_ttl_zero(tmp_path, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(main, "OUTPUT_TTL_MIN", 0)

    old_file = tmp_path / "old.mp4"
    old_file.write_bytes(b"x")
    old_time = time.time() - 3600
    os.utime(old_file, (old_time, old_time))

    main._sweep_stale_media()

    assert old_file.exists()


def test_rate_limit_blocks_after_threshold_then_recovers(monkeypatch):
    import app.main as main

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(main, "_rate_limit_buckets", {})
    monkeypatch.setattr(main, "_RATE_LIMITED_PATHS", {**main._RATE_LIMITED_PATHS, "/agent/plan": 3})

    payload = {
        "prompt": "cut from 4 to 8 seconds",
        "duration_sec": 20.0,
        "sprite_interval_sec": 1.0,
        "total_frames": 20,
        "sheets_count": 1,
    }

    for _ in range(3):
        response = client.post("/agent/plan", json=payload)
        assert response.status_code == 200

    limited = client.post("/agent/plan", json=payload)
    assert limited.status_code == 429
    assert "Retry-After" in limited.headers

    # A different endpoint has its own independent bucket.
    unaffected = client.post(
        "/analyze/token-estimate",
        json={"duration_sec": 10, "interval_sec": 1.0, "columns": 8, "rows": 8, "thumb_width": 256},
    )
    assert unaffected.status_code == 200


def test_rate_limit_bucket_is_keyed_by_client_ip_and_path(monkeypatch):
    import app.main as main

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(main, "_rate_limit_buckets", {})
    monkeypatch.setattr(main, "_RATE_LIMITED_PATHS", {**main._RATE_LIMITED_PATHS, "/agent/summarize": 1})

    payload = {"older_turns": [{"role": "user", "content": "cut the intro"}], "previous_summary": None}

    first = client.post("/agent/summarize", json=payload)
    assert first.status_code == 200
    # Bucketed by (ip, path), not path alone — one busy client on this path
    # shouldn't also throttle a different client, or a different endpoint.
    [(bucket_ip, bucket_path)] = main._rate_limit_buckets.keys()
    assert bucket_path == "/agent/summarize"
    assert isinstance(bucket_ip, str) and bucket_ip

    second = client.post("/agent/summarize", json=payload)
    assert second.status_code == 429
    assert "Retry-After" in second.headers


