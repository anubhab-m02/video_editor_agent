from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app import video_tools


def test_detect_silence_parses_matched_start_end_pairs(monkeypatch, tmp_path):
    stderr = (
        "[silencedetect @ 0x1] silence_start: 2.5\n"
        "[silencedetect @ 0x1] silence_end: 4.1 | silence_duration: 1.6\n"
        "[silencedetect @ 0x1] silence_start: 10.0\n"
        "[silencedetect @ 0x1] silence_end: 12.75 | silence_duration: 2.75\n"
    )
    monkeypatch.setattr(video_tools, "_run_capture_stderr", lambda cmd: stderr)

    result = video_tools.detect_silence(tmp_path / "fake.mp4")

    assert result == [(2.5, 4.1), (10.0, 12.75)]


def test_detect_silence_clips_trailing_silence_to_duration(monkeypatch, tmp_path):
    stderr = "[silencedetect @ 0x1] silence_start: 8.0\n"
    monkeypatch.setattr(video_tools, "_run_capture_stderr", lambda cmd: stderr)
    monkeypatch.setattr(video_tools, "get_duration_sec", lambda path: 10.0)

    result = video_tools.detect_silence(tmp_path / "fake.mp4")

    assert result == [(8.0, 10.0)]


def test_detect_silence_returns_empty_when_no_silence(monkeypatch, tmp_path):
    monkeypatch.setattr(video_tools, "_run_capture_stderr", lambda cmd: "no silence here")

    result = video_tools.detect_silence(tmp_path / "fake.mp4")

    assert result == []
