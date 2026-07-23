"use client";

import type React from "react";
import { create } from "zustand";
import { persist } from "zustand/middleware";

export type TrimRange = { start: number; end: number };
export type SpeedRange = { start: number; end: number; speed: number };
export type FileFingerprint = { name: string; size: number; lastModified: number };

type EditSnapshot = { trimRanges: TrimRange[]; speedRanges: SpeedRange[] };

export function fingerprintsMatch(a: FileFingerprint | null, b: FileFingerprint): boolean {
  return a !== null && a.name === b.name && a.size === b.size && a.lastModified === b.lastModified;
}

export interface VideoStoreState {
  videoSrc: string | null;
  sourceFile: File | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  duration: number;
  currentTime: number;
  isPlaying: boolean;
  isMuted: boolean;
  volume: number;
  trimStart: number;
  trimEnd: number;
  trimRanges: TrimRange[];
  speedRanges: SpeedRange[];
  hasVideo: boolean;
  undoStack: EditSnapshot[];
  redoStack: EditSnapshot[];
  lastFileFingerprint: FileFingerprint | null;
}

export interface VideoStoreActions {
  loadFile: (file: File) => void;
  play: () => void;
  pause: () => void;
  togglePlayPause: () => void;
  seek: (time: number) => void;
  setVolume: (v: number) => void;
  toggleMute: () => void;
  setTrimStart: (t: number) => void;
  setTrimEnd: (t: number) => void;
  setTrimRange: (start: number, end: number) => void;
  setTrimRanges: (ranges: TrimRange[]) => void;
  setSpeedRanges: (ranges: SpeedRange[]) => void;
  requestFullscreen: () => void;
  setCurrentTime: (time: number) => void;
  setDuration: (duration: number) => void;
  setIsPlaying: (isPlaying: boolean) => void;
  undo: () => void;
  redo: () => void;
}

export type VideoStore = VideoStoreState & VideoStoreActions;

const ACCEPTED_TYPES = ["video/mp4", "video/quicktime", "video/webm"];
const EXIT_EPSILON_SEC = 0.02;
const MIN_TRIM_DURATION_SEC = 0.05;
const MIN_SPEED_DURATION_SEC = 0.1;
const DEFAULT_SPEED = 2;
const MAX_HISTORY = 50;

function snapshotEquals(a: EditSnapshot, trimRanges: TrimRange[], speedRanges: SpeedRange[]): boolean {
  return (
    JSON.stringify(a.trimRanges) === JSON.stringify(trimRanges) &&
    JSON.stringify(a.speedRanges) === JSON.stringify(speedRanges)
  );
}

function normalizeTrimRanges(
  ranges: TrimRange[],
  clamp: (time: number) => number
): TrimRange[] {
  const normalized = ranges
    .map((r) => ({
      start: clamp(Math.min(r.start, r.end)),
      end: clamp(Math.max(r.start, r.end)),
    }))
    .filter((r) => r.end - r.start > MIN_TRIM_DURATION_SEC)
    .sort((a, b) => a.start - b.start);

  if (normalized.length <= 1) return normalized;

  const merged: TrimRange[] = [normalized[0]];
  for (let i = 1; i < normalized.length; i += 1) {
    const current = normalized[i];
    const last = merged[merged.length - 1];
    if (current.start <= last.end + MIN_TRIM_DURATION_SEC) {
      last.end = Math.max(last.end, current.end);
    } else {
      merged.push(current);
    }
  }
  return merged;
}

function normalizeSpeedRanges(
  ranges: SpeedRange[],
  clamp: (time: number) => number
): SpeedRange[] {
  return ranges
    .map((r) => ({
      start: clamp(Math.min(r.start, r.end)),
      end: clamp(Math.max(r.start, r.end)),
      speed: Math.max(0.25, Math.min(16, r.speed || DEFAULT_SPEED)),
    }))
    .filter((r) => r.end - r.start > MIN_SPEED_DURATION_SEC)
    .sort((a, b) => a.start - b.start);
}

/**
 * Remove any intervals in `targets` that overlap with `subtract`.
 * Partially-overlapping targets are split or shrunk; fully-covered targets
 * are dropped. Returns a new array (never mutates).
 */
function subtractRanges<T extends { start: number; end: number }>(
  targets: T[],
  subtract: Array<{ start: number; end: number }>,
  minDuration: number
): T[] {
  let result: T[] = [...targets];
  for (const sub of subtract) {
    const next: T[] = [];
    for (const t of result) {
      // No overlap → keep as-is
      if (t.end <= sub.start || t.start >= sub.end) {
        next.push(t);
        continue;
      }
      // Left remnant
      if (t.start < sub.start && sub.start - t.start > minDuration) {
        next.push({ ...t, end: sub.start });
      }
      // Right remnant
      if (t.end > sub.end && t.end - sub.end > minDuration) {
        next.push({ ...t, start: sub.end });
      }
      // Fully covered → dropped (neither remnant added)
    }
    result = next;
  }
  return result;
}

function makeClamp(duration: number) {
  return (time: number) => {
    if (duration <= 0) return Math.max(0, time);
    return Math.max(0, Math.min(time, duration));
  };
}

const videoRef = { current: null } as React.RefObject<HTMLVideoElement | null>;

export const useVideoStore = create<VideoStore>()(
  persist(
    (set, get) => ({
  videoSrc: null,
  sourceFile: null,
  videoRef,
  duration: 0,
  currentTime: 0,
  isPlaying: false,
  isMuted: false,
  volume: 80,
  trimStart: 0,
  trimEnd: 0,
  trimRanges: [],
  speedRanges: [],
  hasVideo: false,
  undoStack: [],
  redoStack: [],
  lastFileFingerprint: null,

  loadFile: (file) => {
    if (!ACCEPTED_TYPES.includes(file.type)) return;
    const prev = get().videoSrc;
    if (prev) URL.revokeObjectURL(prev);
    const src = URL.createObjectURL(file);
    const fingerprint: FileFingerprint = {
      name: file.name,
      size: file.size,
      lastModified: file.lastModified,
    };
    // Same file re-loaded (e.g. after a refresh) -> restore persisted edits.
    // A different/new file always starts clean.
    const state = get();
    const isSameFile = fingerprintsMatch(state.lastFileFingerprint, fingerprint);
    const trimRanges = isSameFile ? state.trimRanges : [];
    const speedRanges = isSameFile ? state.speedRanges : [];
    set({
      videoSrc: src,
      sourceFile: file,
      hasVideo: true,
      isPlaying: false,
      currentTime: 0,
      duration: 0,
      trimStart: trimRanges[0]?.start ?? 0,
      trimEnd: trimRanges[0]?.end ?? 0,
      trimRanges,
      speedRanges,
      undoStack: [],
      redoStack: [],
      lastFileFingerprint: fingerprint,
    });
  },

  play: () => {
    const state = get();
    const video = state.videoRef.current;
    if (!video) return;
    const clamp = makeClamp(state.duration);
    if (state.trimRanges.length > 0) {
      const range = state.trimRanges.find(
        (r) => video.currentTime > r.start && video.currentTime < r.end
      );
      if (range) {
        const next = clamp(range.end + EXIT_EPSILON_SEC);
        video.currentTime = next;
        set({ currentTime: range.end });
      }
    }
    void video.play();
  },

  pause: () => {
    get().videoRef.current?.pause();
  },

  togglePlayPause: () => {
    const state = get();
    const video = state.videoRef.current;
    if (!video) return;
    const clamp = makeClamp(state.duration);
    if (video.paused) {
      if (state.trimRanges.length > 0) {
        const range = state.trimRanges.find(
          (r) => video.currentTime > r.start && video.currentTime < r.end
        );
        if (range) {
          const next = clamp(range.end + EXIT_EPSILON_SEC);
          video.currentTime = next;
          set({ currentTime: range.end });
        }
      }
      void video.play();
    } else {
      video.pause();
    }
  },

  seek: (time) => {
    const state = get();
    const video = state.videoRef.current;
    if (!video) return;
    const clamp = makeClamp(state.duration);
    const clamped = clamp(time);
    const containing = state.trimRanges.find(
      (range) => clamped > range.start && clamped < range.end
    );
    const nextTime = containing
      ? clamped < (containing.start + containing.end) / 2
        ? containing.start
        : clamp(containing.end + EXIT_EPSILON_SEC)
      : clamped;
    video.currentTime = nextTime;
    set({ currentTime: nextTime });
  },

  setVolume: (v) => {
    set((state) => ({
      volume: v,
      isMuted: v > 0 ? false : state.isMuted,
    }));
  },

  toggleMute: () => {
    set((state) => ({ isMuted: !state.isMuted }));
  },

  setTrimRange: (start, end) => {
    const state = get();
    const clamp = makeClamp(state.duration);
    const normalized = normalizeTrimRanges([{ start, end }], clamp);
    const prunedSpeed = subtractRanges(state.speedRanges, normalized, MIN_SPEED_DURATION_SEC);
    const before: EditSnapshot = { trimRanges: state.trimRanges, speedRanges: state.speedRanges };
    const history = snapshotEquals(before, normalized, prunedSpeed)
      ? {}
      : { undoStack: [...state.undoStack, before].slice(-MAX_HISTORY), redoStack: [] };
    if (normalized.length === 0) {
      set({ trimStart: 0, trimEnd: 0, trimRanges: [], speedRanges: prunedSpeed, ...history });
      return;
    }
    set({
      trimStart: normalized[0].start,
      trimEnd: normalized[0].end,
      trimRanges: normalized,
      speedRanges: prunedSpeed,
      ...history,
    });
  },

  setTrimRanges: (ranges) => {
    const state = get();
    const clamp = makeClamp(state.duration);
    const normalized = normalizeTrimRanges(ranges, clamp);
    // Prune speed ranges that overlap with newly-set trim ranges
    const prunedSpeed = subtractRanges(state.speedRanges, normalized, MIN_SPEED_DURATION_SEC);
    const before: EditSnapshot = { trimRanges: state.trimRanges, speedRanges: state.speedRanges };
    const history = snapshotEquals(before, normalized, prunedSpeed)
      ? {}
      : { undoStack: [...state.undoStack, before].slice(-MAX_HISTORY), redoStack: [] };
    if (normalized.length === 0) {
      set({ trimStart: 0, trimEnd: 0, trimRanges: [], speedRanges: prunedSpeed, ...history });
      return;
    }
    set({
      trimStart: normalized[0].start,
      trimEnd: normalized[0].end,
      trimRanges: normalized,
      speedRanges: prunedSpeed,
      ...history,
    });
  },

  setTrimStart: (t) => {
    const state = get();
    state.setTrimRange(t, state.trimEnd);
  },

  setTrimEnd: (t) => {
    const state = get();
    state.setTrimRange(state.trimStart, t);
  },

  requestFullscreen: () => {
    const video = get().videoRef.current;
    if (!video) return;
    if (document.fullscreenElement) {
      void document.exitFullscreen();
    } else {
      void video.requestFullscreen();
    }
  },

  setSpeedRanges: (ranges) => {
    const state = get();
    const clamp = makeClamp(state.duration);
    const normalized = normalizeSpeedRanges(ranges, clamp);
    // Trim has higher priority than speed; clip speed around trimmed gaps.
    const prunedSpeed = subtractRanges(normalized, state.trimRanges, MIN_SPEED_DURATION_SEC);
    const before: EditSnapshot = { trimRanges: state.trimRanges, speedRanges: state.speedRanges };
    const history = snapshotEquals(before, state.trimRanges, prunedSpeed)
      ? {}
      : { undoStack: [...state.undoStack, before].slice(-MAX_HISTORY), redoStack: [] };
    set({
      speedRanges: prunedSpeed,
      ...history,
    });
  },

  setCurrentTime: (time) => set({ currentTime: time }),
  setDuration: (duration) => set({ duration }),
  setIsPlaying: (isPlaying) => set({ isPlaying }),

  undo: () => {
    const state = get();
    if (state.undoStack.length === 0) return;
    const previous = state.undoStack[state.undoStack.length - 1];
    const current: EditSnapshot = { trimRanges: state.trimRanges, speedRanges: state.speedRanges };
    set({
      trimRanges: previous.trimRanges,
      speedRanges: previous.speedRanges,
      trimStart: previous.trimRanges[0]?.start ?? 0,
      trimEnd: previous.trimRanges[0]?.end ?? 0,
      undoStack: state.undoStack.slice(0, -1),
      redoStack: [...state.redoStack, current].slice(-MAX_HISTORY),
    });
  },

  redo: () => {
    const state = get();
    if (state.redoStack.length === 0) return;
    const next = state.redoStack[state.redoStack.length - 1];
    const current: EditSnapshot = { trimRanges: state.trimRanges, speedRanges: state.speedRanges };
    set({
      trimRanges: next.trimRanges,
      speedRanges: next.speedRanges,
      trimStart: next.trimRanges[0]?.start ?? 0,
      trimEnd: next.trimRanges[0]?.end ?? 0,
      undoStack: [...state.undoStack, current].slice(-MAX_HISTORY),
      redoStack: state.redoStack.slice(0, -1),
    });
  },
    }),
    {
      name: "video-editor-session",
      partialize: (state) => ({
        trimRanges: state.trimRanges,
        speedRanges: state.speedRanges,
        lastFileFingerprint: state.lastFileFingerprint,
      }),
    }
  )
);
