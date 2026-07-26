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


def test_detect_scene_changes_parses_pts_time_in_order(monkeypatch, tmp_path):
    stderr = (
        "[Parsed_showinfo_1 @ 0x1] n:   0 pts:    100 pts_time:12.500000 ...\n"
        "[Parsed_showinfo_1 @ 0x1] n:   1 pts:    400 pts_time:47.250000 ...\n"
    )
    monkeypatch.setattr(video_tools, "_run_capture_stderr", lambda cmd: stderr)

    result = video_tools.detect_scene_changes(tmp_path / "fake.mp4")

    assert result == [12.5, 47.25]


def test_detect_scene_changes_caps_at_max_results(monkeypatch, tmp_path):
    stderr = "".join(f"[showinfo] pts_time:{i}.000000 ...\n" for i in range(10))
    monkeypatch.setattr(video_tools, "_run_capture_stderr", lambda cmd: stderr)

    result = video_tools.detect_scene_changes(tmp_path / "fake.mp4", max_results=3)

    assert result == [0.0, 1.0, 2.0]


def test_detect_scene_changes_returns_empty_when_no_cuts(monkeypatch, tmp_path):
    monkeypatch.setattr(video_tools, "_run_capture_stderr", lambda cmd: "nothing detected")

    result = video_tools.detect_scene_changes(tmp_path / "fake.mp4")

    assert result == []
