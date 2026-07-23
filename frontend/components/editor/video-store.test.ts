import { describe, expect, it, vi } from "vitest";

import { useVideoStore } from "./video-store";

// jsdom doesn't implement these; loadFile() needs them.
URL.createObjectURL = vi.fn(() => "blob:mock");
URL.revokeObjectURL = vi.fn();

function resetStore() {
  const state = useVideoStore.getState();
  state.setDuration(10);
  state.setCurrentTime(0);
  state.setIsPlaying(false);
  state.setTrimRanges([]);
  state.videoRef.current = null;
  useVideoStore.setState({ undoStack: [], redoStack: [], lastFileFingerprint: null });
}

describe("video-store", () => {
  it("merges overlapping and adjacent trim ranges", () => {
    resetStore();
    const state = useVideoStore.getState();
    state.setTrimRanges([
      { start: 1, end: 3 },
      { start: 2.98, end: 4 },
      { start: 8, end: 12 },
    ]);

    const ranges = useVideoStore.getState().trimRanges;
    expect(ranges).toHaveLength(2);
    expect(ranges[0].start).toBeCloseTo(1, 2);
    expect(ranges[0].end).toBeCloseTo(4, 2);
    expect(ranges[1].start).toBeCloseTo(8, 2);
    expect(ranges[1].end).toBeCloseTo(10, 2);
  });

  it("seek snaps outside trimmed gap using nearest boundary behavior", () => {
    resetStore();
    const state = useVideoStore.getState();
    const video = {
      currentTime: 0,
      paused: true,
      play: () => Promise.resolve(),
      pause: () => undefined,
      requestFullscreen: () => Promise.resolve(),
      volume: 1,
      muted: false,
    };
    state.videoRef.current = video as unknown as HTMLVideoElement;
    state.setTrimRanges([{ start: 2, end: 4 }]);

    state.seek(2.2);
    expect(video.currentTime).toBeCloseTo(2, 2);

    state.seek(3.8);
    expect(video.currentTime).toBeGreaterThan(4);
  });

  it("normalizes and clamps speed ranges", () => {
    resetStore();
    const state = useVideoStore.getState();
    state.setSpeedRanges([
      { start: 3, end: 1, speed: 2 },   // inverted start/end
      { start: 8, end: 15, speed: 3 },  // exceeds duration (10s)
      { start: 5, end: 5.05, speed: 2 }, // too short (< 0.1s)
    ]);

    const ranges = useVideoStore.getState().speedRanges;
    expect(ranges).toHaveLength(2);
    // Inverted range normalized
    expect(ranges[0].start).toBeCloseTo(1, 2);
    expect(ranges[0].end).toBeCloseTo(3, 2);
    expect(ranges[0].speed).toBe(2);
    // Clamped to duration
    expect(ranges[1].start).toBeCloseTo(8, 2);
    expect(ranges[1].end).toBeCloseTo(10, 2);
    expect(ranges[1].speed).toBe(3);
  });

  it("clamps speed values to [0.25, 16]", () => {
    resetStore();
    const state = useVideoStore.getState();
    state.setSpeedRanges([
      { start: 0, end: 2, speed: 0.1 },   // below min
      { start: 3, end: 5, speed: 20 },    // above max
    ]);

    const ranges = useVideoStore.getState().speedRanges;
    expect(ranges).toHaveLength(2);
    expect(ranges[0].speed).toBe(0.25);
    expect(ranges[1].speed).toBe(16);
  });

  it("splits full-speed range when multiple trims are added", () => {
    resetStore();
    const state = useVideoStore.getState();
    state.setSpeedRanges([{ start: 0, end: 10, speed: 2 }]);
    state.setTrimRanges([
      { start: 3, end: 4 },
      { start: 7, end: 8 },
    ]);

    const ranges = useVideoStore.getState().speedRanges;
    expect(ranges).toEqual([
      { start: 0, end: 3, speed: 2 },
      { start: 4, end: 7, speed: 2 },
      { start: 8, end: 10, speed: 2 },
    ]);
  });

  it("clips incoming speed ranges to avoid trimmed sections", () => {
    resetStore();
    const state = useVideoStore.getState();
    state.setTrimRanges([{ start: 3, end: 7 }]);
    state.setSpeedRanges([{ start: 2, end: 8, speed: 2 }]);

    const ranges = useVideoStore.getState().speedRanges;
    expect(ranges).toEqual([
      { start: 2, end: 3, speed: 2 },
      { start: 7, end: 8, speed: 2 },
    ]);
  });

  it("undo reverts the last trim change and redo reapplies it", () => {
    resetStore();
    const state = useVideoStore.getState();
    state.setTrimRanges([{ start: 1, end: 3 }]);
    state.setTrimRanges([{ start: 1, end: 3 }, { start: 5, end: 6 }]);

    state.undo();
    expect(useVideoStore.getState().trimRanges).toEqual([{ start: 1, end: 3 }]);

    state.undo();
    expect(useVideoStore.getState().trimRanges).toEqual([]);

    state.redo();
    expect(useVideoStore.getState().trimRanges).toEqual([{ start: 1, end: 3 }]);

    state.redo();
    expect(useVideoStore.getState().trimRanges).toEqual([{ start: 1, end: 3 }, { start: 5, end: 6 }]);
  });

  it("undo is a no-op with an empty history stack", () => {
    resetStore();
    const state = useVideoStore.getState();
    state.undo();
    expect(useVideoStore.getState().trimRanges).toEqual([]);
  });

  it("a new edit after undo clears the redo stack", () => {
    resetStore();
    const state = useVideoStore.getState();
    state.setTrimRanges([{ start: 1, end: 3 }]);
    state.undo();
    expect(useVideoStore.getState().redoStack).toHaveLength(1);

    state.setTrimRanges([{ start: 2, end: 4 }]);
    expect(useVideoStore.getState().redoStack).toHaveLength(0);
  });

  it("does not push history for a no-op write of identical ranges", () => {
    resetStore();
    const state = useVideoStore.getState();
    state.setTrimRanges([{ start: 1, end: 3 }]);
    const historyDepth = useVideoStore.getState().undoStack.length;

    state.setTrimRanges([{ start: 1, end: 3 }]);
    expect(useVideoStore.getState().undoStack).toHaveLength(historyDepth);
  });

  it("loadFile resets the undo/redo stacks", () => {
    resetStore();
    const state = useVideoStore.getState();
    state.setTrimRanges([{ start: 1, end: 3 }]);
    expect(useVideoStore.getState().undoStack.length).toBeGreaterThan(0);

    const file = new File([new Blob()], "clip.mp4", { type: "video/mp4" });
    state.loadFile(file);

    expect(useVideoStore.getState().undoStack).toEqual([]);
    expect(useVideoStore.getState().redoStack).toEqual([]);
  });

  it("re-loading the same file (matching name/size/lastModified) restores persisted ranges", () => {
    resetStore();
    const state = useVideoStore.getState();
    const file = new File([new Blob(["x".repeat(10)])], "clip.mp4", {
      type: "video/mp4",
      lastModified: 123456,
    });
    state.loadFile(file);
    useVideoStore.getState().setTrimRanges([{ start: 1, end: 3 }]);

    // Simulate a page refresh: same File re-selected, same identity fields.
    const sameFileAgain = new File([new Blob(["x".repeat(10)])], "clip.mp4", {
      type: "video/mp4",
      lastModified: 123456,
    });
    useVideoStore.getState().loadFile(sameFileAgain);

    expect(useVideoStore.getState().trimRanges).toEqual([{ start: 1, end: 3 }]);
  });

  it("loading a different file does not inherit the previous file's ranges", () => {
    resetStore();
    const state = useVideoStore.getState();
    const file = new File([new Blob(["x".repeat(10)])], "clip.mp4", {
      type: "video/mp4",
      lastModified: 123456,
    });
    state.loadFile(file);
    useVideoStore.getState().setTrimRanges([{ start: 1, end: 3 }]);

    const otherFile = new File([new Blob(["y".repeat(20)])], "other.mp4", {
      type: "video/mp4",
      lastModified: 999,
    });
    useVideoStore.getState().loadFile(otherFile);

    expect(useVideoStore.getState().trimRanges).toEqual([]);
  });
});
