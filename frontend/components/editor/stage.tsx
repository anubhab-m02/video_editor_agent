"use client";

import { useCallback, useEffect, useRef, type DragEvent, type ChangeEvent } from "react";
import { useState } from "react";
import {
    Play,
    Pause,
    Volume2,
    VolumeX,
    Maximize2,
    Upload,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { useVideo } from "./video-context";

function formatTime(seconds: number): string {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(ms).padStart(2, "0")}`;
}

export function Stage() {
    const {
        videoSrc,
        videoRef,
        hasVideo,
        isPlaying,
        isMuted,
        volume,
        currentTime,
        duration,
        loadFile,
        togglePlayPause,
        seek,
        setVolume,
        toggleMute,
        requestFullscreen,
    } = useVideo();

    const fileInputRef = useRef<HTMLInputElement>(null);
    const [isDragOver, setIsDragOver] = useState(false);

    // ── File handling ──────────────────────────────────────────────

    const handleDrop = useCallback(
        (e: DragEvent<HTMLDivElement>) => {
            e.preventDefault();
            setIsDragOver(false);
            const file = e.dataTransfer.files[0];
            if (file) loadFile(file);
        },
        [loadFile]
    );

    const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragOver(true);
    }, []);

    const handleDragLeave = useCallback(() => {
        setIsDragOver(false);
    }, []);

    const handleFileSelect = useCallback(
        (e: ChangeEvent<HTMLInputElement>) => {
            const file = e.target.files?.[0];
            if (file) loadFile(file);
            e.target.value = "";
        },
        [loadFile]
    );

    // Spacebar play/pause (standard video-player convention). Skipped while
    // focus is in a text input/textarea/contenteditable (the chat box) so
    // typing a space there isn't hijacked into toggling playback.
    useEffect(() => {
        function onKeyDown(e: KeyboardEvent) {
            if (e.code !== "Space" && e.key !== " ") return;
            const target = e.target as HTMLElement | null;
            const tag = target?.tagName;
            if (tag === "INPUT" || tag === "TEXTAREA" || target?.isContentEditable) return;
            if (!hasVideo) return;
            e.preventDefault();
            togglePlayPause();
        }
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, [hasVideo, togglePlayPause]);

    // ── Render ─────────────────────────────────────────────────────

    return (
        <div
            className="relative flex flex-1 flex-col overflow-hidden"
            style={{
                background: "linear-gradient(180deg, rgba(24,24,27,0.5) 0%, color-mix(in srgb, var(--color-bg) 60%, transparent) 100%)",
                backdropFilter: "blur(20px)",
            }}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
        >
            {/* Hidden file input */}
            <input
                ref={fileInputRef}
                type="file"
                accept="video/mp4,video/quicktime,video/webm"
                className="hidden"
                onChange={handleFileSelect}
            />

            {/* Video Preview Area */}
            <div className="relative flex w-full flex-1 items-center justify-center p-4">
                {hasVideo ? (
                    <div
                        className="relative h-full w-full overflow-hidden rounded-lg"
                        style={{
                            background: "rgb(0,0,0)",
                            boxShadow: "0 0 0 1px var(--color-border), 0 8px 32px rgba(0,0,0,0.5), 0 2px 8px rgba(0,0,0,0.3)",
                        }}
                    >                        <video
                            ref={videoRef}
                            src={videoSrc!}
                            className="absolute inset-0 h-full w-full object-contain"
                            playsInline
                            preload="metadata"
                        />
                    </div>
                ) : (
                    /* Upload prompt */
                    <button
                        onClick={() => fileInputRef.current?.click()}
                        className={`group relative flex h-full w-full cursor-pointer flex-col items-center justify-center gap-4 rounded-xl border transition-all duration-200 ${isDragOver ? "border-primary/60 bg-primary/5" : "border-zinc-700/50 hover:border-zinc-600/60 hover:bg-white/[0.02]"}`}
                        style={{
                            background: isDragOver ? undefined : "linear-gradient(180deg, rgba(39,39,42,0.3) 0%, rgba(24,24,27,0.4) 100%)",
                            boxShadow: "var(--inset-highlight), 0 4px 24px rgba(0,0,0,0.2)",
                        }}
                    >
                        <div className={`flex h-16 w-16 items-center justify-center rounded-2xl transition-colors duration-200 ${isDragOver ? "bg-primary/20" : "bg-zinc-800 group-hover:bg-primary/20"}`}>
                            <Upload className={`h-7 w-7 transition-colors duration-200 ${isDragOver ? "text-primary" : "text-zinc-400 group-hover:text-primary"}`} />
                        </div>
                        <div className="text-center">
                            <p className="text-sm font-medium text-zinc-300">
                                {isDragOver ? "Drop to upload" : "Drop your video here"}
                            </p>
                            <p className="mt-1 text-xs text-zinc-400">
                                or click to browse files
                            </p>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="rounded-md bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-400">MP4</span>
                            <span className="rounded-md bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-400">MOV</span>
                            <span className="rounded-md bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-400">WEBM</span>
                        </div>
                    </button>
                )}
            </div>

            {/* Transport Controls */}
            <div
                className={`flex w-full shrink-0 flex-col gap-1.5 px-4 py-2 transition-opacity duration-200 ${hasVideo ? "opacity-100" : "pointer-events-none opacity-30"}`}
                style={{
                    borderTop: "1px solid var(--color-border)",
                    background: "linear-gradient(180deg, rgba(39,39,42,0.4) 0%, rgba(24,24,27,0.5) 100%)",
                }}
            >
                {/* Seek bar */}
                <Slider
                    value={[currentTime]}
                    max={duration || 1}
                    step={0.01}
                    onValueChange={(v) => seek(v[0])}
                    className="w-full cursor-pointer"
                />

                {/* Controls row */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1">
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={togglePlayPause}
                            className="h-7 w-7 text-zinc-300 hover:text-white"
                        >
                            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                        </Button>

                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={toggleMute}
                            className="h-7 w-7 text-zinc-300 hover:text-white"
                        >
                            {isMuted || volume === 0 ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
                        </Button>

                        <Slider
                            value={isMuted ? [0] : [volume]}
                            max={100}
                            step={1}
                            onValueChange={(v) => setVolume(v[0])}
                            className="w-20"
                        />
                    </div>

                    {/* Timecode */}
                    <div className="font-mono text-xs tabular-nums text-zinc-400">
                        <span className="text-zinc-200">{formatTime(currentTime)}</span>
                        <span className="mx-1 text-zinc-600">/</span>
                        <span>{formatTime(duration)}</span>
                    </div>

                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={requestFullscreen}
                        className="h-7 w-7 text-zinc-400 hover:text-white"
                    >
                        <Maximize2 className="h-3.5 w-3.5" />
                    </Button>
                </div>
            </div>
        </div>
    );
}
