"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import {
    Plus,
    ZoomIn,
    ZoomOut,
    Scissors,
    Trash2,
    RotateCcw,
    FastForward,
    Undo2,
    Redo2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import { useVideo } from "./video-context";

const TOOLBAR_ITEMS = [
    { icon: Plus, label: "Add segment", shortcut: "A" },
    { icon: ZoomIn, label: "Zoom in", shortcut: "+" },
    { icon: ZoomOut, label: "Zoom out", shortcut: "-" },
    { icon: Scissors, label: "Trim", shortcut: "T" },
    { icon: Trash2, label: "Delete", shortcut: "⌫" },
    { icon: Undo2, label: "Undo", shortcut: "⌘Z" },
    { icon: Redo2, label: "Redo", shortcut: "⌘⇧Z" },
    { icon: RotateCcw, label: "Reset timeline", shortcut: "" },
] as const;

// Keyboard nudging for the active trim/speed widget (a11y: mouse-only drag isn't
// an acceptable sole input path). Shift = resize instead of move; Cmd/Ctrl = larger step.
const NUDGE_STEP_SEC = 0.1;
const NUDGE_STEP_LARGE_SEC = 1;

type TrimWidget = { id: string; startTime: number; duration: number };
type SpeedWidget = { id: string; startTime: number; duration: number; speed: number };

function areRangesEquivalent(
    widgets: TrimWidget[],
    ranges: Array<{ start: number; end: number }>
): boolean {
    if (widgets.length !== ranges.length) return false;
    const sortedWidgets = [...widgets].sort((a, b) => a.startTime - b.startTime);
    const sortedRanges = [...ranges].sort((a, b) => a.start - b.start);
    return sortedWidgets.every((w, i) => {
        const r = sortedRanges[i];
        const startDiff = Math.abs(w.startTime - r.start);
        const endDiff = Math.abs(w.startTime + w.duration - r.end);
        return startDiff < 0.02 && endDiff < 0.02;
    });
}

/** Generate ruler tick marks with major, minor, and micro intervals */
function generateTicks(visibleDuration: number, zoom: number) {
    let majorInterval: number;
    let minorInterval: number;
    let microInterval: number;

    if (zoom <= 0.25) {
        majorInterval = 10;
        minorInterval = 5;
        microInterval = 1;
    } else if (zoom <= 0.5) {
        majorInterval = 5;
        minorInterval = 1;
        microInterval = 0.5;
    } else if (zoom <= 1) {
        majorInterval = 1;
        minorInterval = 0.5;
        microInterval = 0.25;
    } else if (zoom <= 2) {
        majorInterval = 1;
        minorInterval = 0.25;
        microInterval = 0.1;
    } else {
        majorInterval = 0.5;
        minorInterval = 0.1;
        microInterval = 0.05;
    }

    const majorTicks: number[] = [];
    const minorTicks: number[] = [];
    const microTicks: number[] = [];

    const EPS = 0.0001;
    for (let t = 0; t <= visibleDuration + EPS; t += microInterval) {
        const time = Math.round(t * 10000) / 10000; // avoid float drift
        if (time > visibleDuration) break;
        const isMajor =
            Math.abs(time % majorInterval) < EPS ||
            Math.abs(time % majorInterval - majorInterval) < EPS;
        const isMinor =
            Math.abs(time % minorInterval) < EPS ||
            Math.abs(time % minorInterval - minorInterval) < EPS;
        if (isMajor) {
            majorTicks.push(time);
        } else if (isMinor) {
            minorTicks.push(time);
        } else {
            microTicks.push(time);
        }
    }

    return { majorTicks, minorTicks, microTicks };
}

function formatRulerTime(seconds: number, zoom: number): string {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    // If zoomed in significantly, show one decimal place for "smart" precision
    if (zoom > 1.5) {
        const ms = Math.floor((seconds % 1) * 10);
        return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${ms}`;
    }
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function Timeline() {
    const {
        duration,
        currentTime,
        hasVideo,
        trimRanges,
        speedRanges,
        seek,
        setTrimRanges,
        setSpeedRanges,
        undo,
        redo,
    } = useVideo();

    const [zoom, setZoom] = useState(1);
    const [isAddSegmentMenuOpen, setIsAddSegmentMenuOpen] = useState(false);
    const [trimWidgets, setTrimWidgets] = useState<TrimWidget[]>([]);
    const [speedWidgets, setSpeedWidgets] = useState<SpeedWidget[]>([]);
    const [activeTrimId, setActiveTrimId] = useState<string | null>(null);
    const [activeSpeedId, setActiveSpeedId] = useState<string | null>(null);
    const [isDraggingPlayhead, setIsDraggingPlayhead] = useState(false);
    const [dragState, setDragState] = useState<{
        type: "move" | "resize-start" | "resize-end";
        widgetKind: "trim" | "speed";
        widgetId: string;
        startX: number;
        initialStartTime: number;
        initialDuration: number;
    } | null>(null);
    const [containerWidth, setContainerWidth] = useState(0);
    const trackRef = useRef<HTMLDivElement>(null);
    const trimWidgetsRef = useRef<TrimWidget[]>([]);
    const speedWidgetsRef = useRef<SpeedWidget[]>([]);
    const activeTrimIdRef = useRef<string | null>(null);
    const activeSpeedIdRef = useRef<string | null>(null);
    const hasMovedDuringDragRef = useRef(false);
    const suppressNextTrackClickRef = useRef(false);
    const isLocalEditRef = useRef(false);
    const isLocalSpeedEditRef = useRef(false);
    const toolbarActionRef = useRef<(label: string) => void>(() => undefined);

    const commitTrimWidgetsToContext = useCallback(
        (widgets: TrimWidget[]) => {
            isLocalEditRef.current = true;
            setTrimRanges(
                widgets.map((w) => ({
                    start: w.startTime,
                    end: w.startTime + w.duration,
                }))
            );
        },
        [setTrimRanges]
    );

    const commitSpeedWidgetsToContext = useCallback(
        (widgets: SpeedWidget[]) => {
            isLocalSpeedEditRef.current = true;
            setSpeedRanges(
                widgets.map((w) => ({
                    start: w.startTime,
                    end: w.startTime + w.duration,
                    speed: w.speed,
                }))
            );
        },
        [setSpeedRanges]
    );

    useEffect(() => {
        trimWidgetsRef.current = trimWidgets;
    }, [trimWidgets]);

    useEffect(() => {
        speedWidgetsRef.current = speedWidgets;
    }, [speedWidgets]);

    useEffect(() => {
        activeTrimIdRef.current = activeTrimId;
    }, [activeTrimId]);

    useEffect(() => {
        activeSpeedIdRef.current = activeSpeedId;
    }, [activeSpeedId]);

    useEffect(() => {
        if (isLocalEditRef.current) {
            isLocalEditRef.current = false;
            return;
        }
        if (areRangesEquivalent(trimWidgetsRef.current, trimRanges)) return;
        const nextWidgets: TrimWidget[] = trimRanges.map((range) => ({
            id: crypto.randomUUID(),
            startTime: range.start,
            duration: Math.max(0.05, range.end - range.start),
        }));
        setTrimWidgets(nextWidgets);
        if (!nextWidgets.some((w) => w.id === activeTrimIdRef.current)) {
            setActiveTrimId(nextWidgets[0]?.id ?? null);
        }
    }, [trimRanges]);

    // Sync speed widgets from store
    useEffect(() => {
        if (isLocalSpeedEditRef.current) {
            isLocalSpeedEditRef.current = false;
            return;
        }
        const nextWidgets: SpeedWidget[] = speedRanges.map((range) => ({
            id: crypto.randomUUID(),
            startTime: range.start,
            duration: Math.max(0.1, range.end - range.start),
            speed: range.speed,
        }));
        setSpeedWidgets(nextWidgets);
        if (!nextWidgets.some((w) => w.id === activeSpeedIdRef.current)) {
            setActiveSpeedId(nextWidgets[0]?.id ?? null);
        }
    }, [speedRanges]);

    // Track container width so the timeline always fills the available space
    useEffect(() => {
        const el = trackRef.current;
        if (!el) return;
        const ro = new ResizeObserver(([entry]) => {
            setContainerWidth(entry.contentRect.width);
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    // Use real duration or a sensible fallback (30s) for the empty state
    const effectiveDuration = duration > 0 ? duration : 30;
    // At zoom=1, the video clip fills the entire container width (48px padding for the track)
    const trackPadding = 48;
    const basePxPerSecond = containerWidth > trackPadding
        ? (containerWidth - trackPadding) / effectiveDuration
        : 40;
    const pxPerSecond = basePxPerSecond * zoom;
    const videoTrackWidth = effectiveDuration * pxPerSecond;
    // Content width: at least the container, or wider when zoomed in
    const contentWidth = Math.max(videoTrackWidth + trackPadding, containerWidth);
    // Extend ruler ticks to cover the full visible width
    const visibleDurationInSeconds = contentWidth / pxPerSecond;
    const ticks = generateTicks(visibleDurationInSeconds, zoom);

    const posToTime = useCallback(
        (clientX: number) => {
            if (!trackRef.current) return 0;
            const rect = trackRef.current.getBoundingClientRect();
            // Important: subtract rect.left + 24 to account for both viewport offset and track padding
            const x = clientX - rect.left + trackRef.current.scrollLeft - 24;
            return Math.max(0, Math.min(effectiveDuration, x / pxPerSecond));
        },
        [pxPerSecond, effectiveDuration]
    );

    /* Handle global mouse move/up for TRIM WIDGET dragging/resizing */
    useEffect(() => {
        if (!dragState) return;

        const handleMouseMove = (e: PointerEvent) => {
            const deltaX = e.clientX - dragState.startX;
            if (Math.abs(deltaX) > 1) {
                hasMovedDuringDragRef.current = true;
            }
            const deltaSeconds = deltaX / pxPerSecond;

            if (dragState.type === "move") {
                let newStartTime = dragState.initialStartTime + deltaSeconds;
                if (newStartTime < 0) newStartTime = 0;
                if (duration && newStartTime + dragState.initialDuration > duration) {
                    newStartTime = duration - dragState.initialDuration;
                }

                if (dragState.widgetKind === "trim") {
                    setTrimWidgets((prev) =>
                        prev.map((widget) =>
                            widget.id === dragState.widgetId
                                ? { ...widget, startTime: newStartTime }
                                : widget
                        )
                    );
                } else {
                    setSpeedWidgets((prev) =>
                        prev.map((widget) =>
                            widget.id === dragState.widgetId
                                ? { ...widget, startTime: newStartTime }
                                : widget
                        )
                    );
                }
            } else if (dragState.type === "resize-start") {
                let newStartTime = dragState.initialStartTime + deltaSeconds;
                let newDuration = dragState.initialDuration - deltaSeconds;

                if (newDuration < 0.5) {
                    newDuration = 0.5;
                    newStartTime = dragState.initialStartTime + dragState.initialDuration - 0.5;
                }
                if (newStartTime < 0) {
                    newStartTime = 0;
                    newDuration = dragState.initialStartTime + dragState.initialDuration;
                }

                if (dragState.widgetKind === "trim") {
                    setTrimWidgets((prev) =>
                        prev.map((widget) =>
                            widget.id === dragState.widgetId
                                ? { ...widget, startTime: newStartTime, duration: newDuration }
                                : widget
                        )
                    );
                } else {
                    setSpeedWidgets((prev) =>
                        prev.map((widget) =>
                            widget.id === dragState.widgetId
                                ? { ...widget, startTime: newStartTime, duration: newDuration }
                                : widget
                        )
                    );
                }
            } else if (dragState.type === "resize-end") {
                if (dragState.widgetKind === "trim") {
                    setTrimWidgets((prev) =>
                        prev.map((widgetItem) => {
                            if (widgetItem.id !== dragState.widgetId) return widgetItem;
                            let newDuration = dragState.initialDuration + deltaSeconds;
                            if (newDuration < 0.5) newDuration = 0.5;
                            if (duration && widgetItem.startTime + newDuration > duration) {
                                newDuration = duration - widgetItem.startTime;
                            }
                            return { ...widgetItem, duration: newDuration };
                        })
                    );
                } else {
                    setSpeedWidgets((prev) =>
                        prev.map((widgetItem) => {
                            if (widgetItem.id !== dragState.widgetId) return widgetItem;
                            let newDuration = dragState.initialDuration + deltaSeconds;
                            if (newDuration < 0.5) newDuration = 0.5;
                            if (duration && widgetItem.startTime + newDuration > duration) {
                                newDuration = duration - widgetItem.startTime;
                            }
                            return { ...widgetItem, duration: newDuration };
                        })
                    );
                }
            }
        };

        const handleMouseUp = () => {
            if (hasMovedDuringDragRef.current) {
                suppressNextTrackClickRef.current = true;
                hasMovedDuringDragRef.current = false;
            }
            if (dragState.widgetKind === "trim") {
                commitTrimWidgetsToContext(trimWidgetsRef.current);
            } else {
                commitSpeedWidgetsToContext(speedWidgetsRef.current);
            }
            setDragState(null);
        };

        window.addEventListener("pointermove", handleMouseMove);
        window.addEventListener("pointerup", handleMouseUp);

        return () => {
            window.removeEventListener("pointermove", handleMouseMove);
            window.removeEventListener("pointerup", handleMouseUp);
        };
    }, [dragState, pxPerSecond, duration, commitTrimWidgetsToContext, commitSpeedWidgetsToContext]);

    /* Dragging logic — uses global mouse events for smooth drag even outside the track */
    useEffect(() => {
        if (!isDraggingPlayhead) return;

        function onMouseMove(e: PointerEvent) {
            const time = posToTime(e.clientX);
            seek(time);
        }

        function onMouseUp() {
            setIsDraggingPlayhead(false);
        }

        window.addEventListener("pointermove", onMouseMove);
        window.addEventListener("pointerup", onMouseUp);
        return () => {
            window.removeEventListener("pointermove", onMouseMove);
            window.removeEventListener("pointerup", onMouseUp);
        };
    }, [
        isDraggingPlayhead,
        posToTime,
        seek,
    ]);

    /** Auto-scroll to keep the playhead in view during playback */
    useEffect(() => {
        if (!trackRef.current || isDraggingPlayhead) return;
        const container = trackRef.current;
        const playheadX = currentTime * pxPerSecond + 24;
        const { scrollLeft, clientWidth } = container;

        // If playhead is outside the visible range, scroll to it
        if (playheadX < scrollLeft + 40 || playheadX > scrollLeft + clientWidth - 40) {
            container.scrollTo({ left: playheadX - clientWidth / 3, behavior: "smooth" });
        }
    }, [currentTime, pxPerSecond, isDraggingPlayhead]);

    function handleTrackClick(e: React.MouseEvent) {
        if (suppressNextTrackClickRef.current) {
            suppressNextTrackClickRef.current = false;
            return;
        }
        if (!hasVideo) return;
        const time = posToTime(e.clientX);
        seek(time);
    }

    function handleZoomIn() {
        setZoom((z) => Math.min(4, z * 1.25));
    }

    function handleZoomOut() {
        setZoom((z) => Math.max(0.25, z / 1.25));
    }

    function addTrimAtPlayhead() {
        const startTime = Math.max(
            0,
            Math.min(currentTime, Math.max(0, effectiveDuration - 0.5))
        );
        const segmentDuration = Math.min(
            2,
            Math.max(0.5, effectiveDuration - startTime)
        );
        const nextWidget: TrimWidget = {
            id: crypto.randomUUID(),
            startTime,
            duration: segmentDuration,
        };
        const next = [...trimWidgetsRef.current, nextWidget];
        setTrimWidgets(next);
        commitTrimWidgetsToContext(next);
        setActiveTrimId(nextWidget.id);
    }

    function addSpeedAtPlayhead() {
        const startTime = Math.max(
            0,
            Math.min(currentTime, Math.max(0, effectiveDuration - 0.5))
        );
        const segmentDuration = Math.min(
            2,
            Math.max(0.5, effectiveDuration - startTime)
        );
        const nextWidget: SpeedWidget = {
            id: crypto.randomUUID(),
            startTime,
            duration: segmentDuration,
            speed: 2,
        };
        const next = [...speedWidgetsRef.current, nextWidget];
        setSpeedWidgets(next);
        commitSpeedWidgetsToContext(next);
        setActiveSpeedId(nextWidget.id);
    }

    function handleToolbarAction(label: string) {
        if (label === "Add segment") {
            setIsAddSegmentMenuOpen(!isAddSegmentMenuOpen);
            return;
        }

        switch (label) {
            case "Zoom in":
                handleZoomIn();
                break;
            case "Zoom out":
                handleZoomOut();
                break;
            case "Trim":
                addTrimAtPlayhead();
                setIsAddSegmentMenuOpen(false);
                break;
            case "Speed Up":
                addSpeedAtPlayhead();
                setIsAddSegmentMenuOpen(false);
                break;
            case "Delete": {
                if (activeTrimId) {
                    const next = trimWidgets.filter((w) => w.id !== activeTrimId);
                    setTrimWidgets(next);
                    commitTrimWidgetsToContext(next);
                    setActiveTrimId(null);
                }
                if (activeSpeedId) {
                    const next = speedWidgets.filter((w) => w.id !== activeSpeedId);
                    setSpeedWidgets(next);
                    commitSpeedWidgetsToContext(next);
                    setActiveSpeedId(null);
                }
                break;
            }
            case "Undo":
                undo();
                break;
            case "Redo":
                redo();
                break;
            case "Reset timeline":
                setZoom(1);
                setTrimWidgets([]);
                setSpeedWidgets([]);
                setActiveTrimId(null);
                setActiveSpeedId(null);
                setTrimRanges([]);
                setSpeedRanges([]);
                break;
        }
    }

    useEffect(() => {
        toolbarActionRef.current = handleToolbarAction;
    });

    useEffect(() => {
        function isTypingTarget(target: EventTarget | null): boolean {
            if (!(target instanceof HTMLElement)) return false;
            const tag = target.tagName.toLowerCase();
            return (
                tag === "input" ||
                tag === "textarea" ||
                tag === "select" ||
                Boolean(target.closest("[contenteditable='true']"))
            );
        }

        function onKeyDown(e: KeyboardEvent) {
            if (!hasVideo || isTypingTarget(e.target)) return;

            const key = e.key.toLowerCase();
            if ((e.metaKey || e.ctrlKey) && key === "z") {
                e.preventDefault();
                if (e.shiftKey) {
                    redo();
                } else {
                    undo();
                }
                return;
            }
            if (key === "a") {
                e.preventDefault();
                toolbarActionRef.current("Add segment");
                return;
            }
            if (e.key === "+" || (e.key === "=" && e.shiftKey)) {
                e.preventDefault();
                toolbarActionRef.current("Zoom in");
                return;
            }
            if (e.key === "-" || e.key === "_") {
                e.preventDefault();
                toolbarActionRef.current("Zoom out");
                return;
            }
            if (key === "t") {
                e.preventDefault();
                toolbarActionRef.current("Trim");
                return;
            }
            if (e.key === "Backspace" || e.key === "Delete") {
                e.preventDefault();
                toolbarActionRef.current("Delete");
                return;
            }
            if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
                const activeId = activeTrimIdRef.current ?? activeSpeedIdRef.current;
                if (!activeId) return;
                e.preventDefault();
                const isTrim = activeTrimIdRef.current === activeId;
                const step =
                    (e.metaKey || e.ctrlKey ? NUDGE_STEP_LARGE_SEC : NUDGE_STEP_SEC) *
                    (e.key === "ArrowLeft" ? -1 : 1);
                const nudge = <T extends TrimWidget>(w: T): T => {
                    if (w.id !== activeId) return w;
                    if (e.shiftKey) {
                        let newDuration = Math.max(0.5, w.duration + step);
                        if (duration && w.startTime + newDuration > duration) {
                            newDuration = duration - w.startTime;
                        }
                        return { ...w, duration: newDuration };
                    }
                    let newStartTime = Math.max(0, w.startTime + step);
                    if (duration && newStartTime + w.duration > duration) {
                        newStartTime = duration - w.duration;
                    }
                    return { ...w, startTime: newStartTime };
                };
                if (isTrim) {
                    const next = trimWidgetsRef.current.map(nudge);
                    setTrimWidgets(next);
                    commitTrimWidgetsToContext(next);
                } else {
                    const next = speedWidgetsRef.current.map(nudge);
                    setSpeedWidgets(next);
                    commitSpeedWidgetsToContext(next);
                }
            }
        }

        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, [hasVideo, undo, redo, duration, commitTrimWidgetsToContext, commitSpeedWidgetsToContext]);

    return (
        <TooltipProvider delayDuration={200}>
            <div
                className={`flex h-[220px] shrink-0 flex-col transition-opacity duration-200 ${hasVideo ? "opacity-100" : "opacity-40 pointer-events-none"}`}
                style={{
                    borderTop: "1px solid rgba(255,255,255,0.06)",
                    background: "linear-gradient(180deg, rgba(24,24,27,0.8) 0%, rgba(9,9,11,0.9) 100%)",
                    backdropFilter: "blur(20px)",
                }}
            >
                {/* Toolbar */}
                <div
                    className="flex items-center gap-1 px-3 py-1.5"
                    style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}
                >
                    {TOOLBAR_ITEMS.map((item, i) => (
                        <div key={item.label} className="relative flex items-center">
                            {i === 5 && (
                                <Separator orientation="vertical" className="mx-1 h-5 bg-zinc-800" />
                            )}

                            {/* Detailed "Add segment" popover menu */}
                            {item.label === "Add segment" && isAddSegmentMenuOpen && (
                                <div className="absolute bottom-full left-0 mb-2 z-50 min-w-[140px] flex flex-col gap-1 rounded-lg border border-zinc-800 bg-zinc-950 p-1 shadow-xl animate-in fade-in slide-in-from-bottom-2 duration-200">
                                    <Button
                                        variant="ghost"
                                        size="sm"
                                        className="h-9 w-full justify-start gap-2 px-2 text-zinc-100 hover:bg-zinc-800 hover:text-white"
                                        onClick={() => {
                                            addTrimAtPlayhead();
                                            setIsAddSegmentMenuOpen(false);
                                        }}
                                    >
                                        <Scissors className="h-4 w-4 text-zinc-400" />
                                        <span>Trim</span>
                                    </Button>
                                    <Button
                                        variant="ghost"
                                        size="sm"
                                        className="h-9 w-full justify-start gap-2 px-2 text-zinc-100 hover:bg-zinc-800 hover:text-white"
                                        onClick={() => {
                                            addSpeedAtPlayhead();
                                            setIsAddSegmentMenuOpen(false);
                                        }}
                                    >
                                        <FastForward className="h-4 w-4 text-zinc-400" />
                                        <span>Speed Up</span>
                                    </Button>
                                </div>
                            )}

                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className={`h-7 w-7 text-zinc-400 hover:text-zinc-200 ${item.label === "Add segment" && isAddSegmentMenuOpen ? "bg-zinc-800 text-zinc-100" : ""}`}
                                        onClick={() => handleToolbarAction(item.label)}
                                    >
                                        <item.icon className="h-3.5 w-3.5" />
                                    </Button>
                                </TooltipTrigger>
                                <TooltipContent side="bottom" className="bg-zinc-800 text-zinc-200">
                                    <span>{item.label}</span>
                                    {item.shortcut && (
                                        <span className="ml-2 text-zinc-400">{item.shortcut}</span>
                                    )}
                                </TooltipContent>
                            </Tooltip>
                        </div>
                    ))}

                    {/* Zoom indicator */}
                    <div className="ml-auto font-mono text-[11px] text-zinc-400">
                        {Math.round(zoom * 100)}%
                    </div>
                </div>

                {/* Timeline track area */}
                <div
                    ref={trackRef}
                    className="scrollbar-thin relative flex-1 overflow-x-auto overflow-y-hidden"
                    onClick={handleTrackClick}
                >
                    <div
                        className="relative min-h-full flex flex-col"
                        style={{ width: `${contentWidth}px` }}
                    >
                        {/* Track */}
                        <div className="relative px-6 py-3 flex-1 flex flex-col">
                            {/* Track bar */}
                            <div
                                className="relative flex-1 rounded-lg mt-2 select-none"
                                style={{
                                    background: "linear-gradient(180deg, rgba(39,39,42,0.7) 0%, rgba(24,24,27,0.8) 100%)",
                                    boxShadow: "inset 0 1px 0 rgba(255,255,255,0.04), inset 0 -1px 0 rgba(0,0,0,0.3), 0 1px 3px rgba(0,0,0,0.4)",
                                }}
                            >
                                {/* Video clip block */}
                                {hasVideo && (
                                    <div
                                        className="absolute inset-y-0 left-0 overflow-visible rounded-lg"
                                        style={{
                                            width: `${videoTrackWidth}px`,
                                            background: "linear-gradient(180deg, rgba(16,185,129,0.12) 0%, rgba(16,185,129,0.06) 40%, rgba(16,185,129,0.10) 100%)",
                                            boxShadow: "inset 0 1px 0 rgba(52,211,153,0.15), inset 0 -1px 0 rgba(52,211,153,0.08)",
                                        }}
                                    >
                                        {/* Left edge cap — outside the frame */}
                                        <div
                                            className="absolute top-0 bottom-0 w-[5px] rounded-l-lg"
                                            style={{
                                                left: "-5px",
                                                background: "linear-gradient(180deg, rgb(52,211,153) 0%, rgb(16,185,129) 50%, rgb(5,150,105) 100%)",
                                                boxShadow: "inset -1px 0 0 rgba(255,255,255,0.15)",
                                            }}
                                        />
                                        {/* Right edge cap — outside the frame */}
                                        <div
                                            className="absolute top-0 bottom-0 w-[5px] rounded-r-lg"
                                            style={{
                                                right: "-5px",
                                                background: "linear-gradient(180deg, rgb(52,211,153) 0%, rgb(16,185,129) 50%, rgb(5,150,105) 100%)",
                                                boxShadow: "inset 1px 0 0 rgba(255,255,255,0.15)",
                                            }}
                                        />

                                        {/* Subtle top border line */}
                                        <div className="absolute top-0 left-0 right-0 h-px bg-emerald-400/20" />
                                        {/* Subtle bottom border line */}
                                        <div className="absolute bottom-0 left-0 right-0 h-px bg-emerald-400/10" />

                                        {/* Micro Ticks — finest tier */}
                                        {ticks.microTicks.map((t) => (
                                            <div
                                                key={`micro-${t}`}
                                                className="absolute top-0 w-px"
                                                style={{
                                                    left: `${t * pxPerSecond}px`,
                                                    height: "10px",
                                                    background: "rgba(161,161,170,0.22)",
                                                }}
                                            />
                                        ))}

                                        {/* Minor Ticks — medium tier */}
                                        {ticks.minorTicks.map((t) => (
                                            <div
                                                key={`minor-${t}`}
                                                className="absolute top-0 w-px"
                                                style={{
                                                    left: `${t * pxPerSecond}px`,
                                                    height: "18px",
                                                    background: "rgba(161,161,170,0.4)",
                                                }}
                                            />
                                        ))}

                                        {/* Major Ticks & Labels — tallest tier */}
                                        {ticks.majorTicks.map((t) => (
                                            <div
                                                key={`major-${t}`}
                                                className="absolute top-0 flex flex-col items-center"
                                                style={{ left: `${t * pxPerSecond}px` }}
                                            >
                                                {/* Major Tick */}
                                                <div
                                                    className="w-px"
                                                    style={{
                                                        height: "28px",
                                                        background: "linear-gradient(180deg, rgba(161,161,170,0.8) 0%, rgba(161,161,170,0.25) 100%)",
                                                    }}
                                                />
                                                {/* Label */}
                                                <span
                                                    className="absolute top-[30px] -translate-x-1/2 whitespace-nowrap font-mono text-[9px] font-medium tracking-wider"
                                                    style={{ color: "rgba(161,161,170,0.6)" }}
                                                >
                                                    {formatRulerTime(t, zoom)}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                {/* Removed segment masks (non-playable in preview) */}
                                {hasVideo &&
                                    trimWidgets.map((widget) => (
                                        <div
                                            key={`mask-${widget.id}`}
                                            className="absolute inset-y-0 z-[9] rounded-md bg-red-950/30"
                                            style={{
                                                left: `${widget.startTime * pxPerSecond}px`,
                                                width: `${widget.duration * pxPerSecond}px`,
                                                backgroundImage: "repeating-linear-gradient(135deg, transparent, transparent 4px, rgba(239,68,68,0.08) 4px, rgba(239,68,68,0.08) 5px)",
                                            }}
                                        />
                                    ))}

                                {/* Trim Widget Overlays */}
                                {hasVideo &&
                                    trimWidgets.map((widget) => {
                                        const isActive = widget.id === activeTrimId;
                                        return (
                                            <div
                                                key={widget.id}
                                                tabIndex={0}
                                                role="slider"
                                                aria-label={`Trim segment, ${widget.startTime.toFixed(1)}s to ${(widget.startTime + widget.duration).toFixed(1)}s. Arrow keys move, shift+arrow resizes.`}
                                                aria-valuenow={widget.startTime}
                                                onFocus={() => {
                                                    setActiveTrimId(widget.id);
                                                    setActiveSpeedId(null);
                                                }}
                                                className={`absolute inset-y-0 z-10 flex items-center justify-center rounded-lg cursor-move group transition-shadow duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-rose-400 ${isActive ? "shadow-[0_0_0_1px_rgba(244,63,94,0.6),0_0_12px_-2px_rgba(244,63,94,0.3)]" : "shadow-[0_0_0_1px_rgba(244,63,94,0.35)]"}`}
                                                style={{
                                                    left: `${widget.startTime * pxPerSecond}px`,
                                                    width: `${widget.duration * pxPerSecond}px`,
                                                    background: isActive
                                                        ? "linear-gradient(135deg, rgba(244,63,94,0.25) 0%, rgba(251,113,133,0.18) 50%, rgba(244,63,94,0.22) 100%)"
                                                        : "linear-gradient(135deg, rgba(244,63,94,0.15) 0%, rgba(251,113,133,0.10) 50%, rgba(244,63,94,0.12) 100%)",
                                                    backdropFilter: "blur(6px)",
                                                }}
                                                onPointerDown={(e) => {
                                                    e.stopPropagation();
                                                    hasMovedDuringDragRef.current = false;
                                                    setActiveTrimId(widget.id);
                                                    setActiveSpeedId(null);
                                                    setDragState({
                                                        type: "move",
                                                        widgetKind: "trim",
                                                        widgetId: widget.id,
                                                        startX: e.clientX,
                                                        initialStartTime: widget.startTime,
                                                        initialDuration: widget.duration,
                                                    });
                                                }}
                                            >
                                                {/* Diagonal stripe pattern overlay */}
                                                <div
                                                    className="absolute inset-0 rounded-lg opacity-30 pointer-events-none"
                                                    style={{
                                                        backgroundImage: "repeating-linear-gradient(135deg, transparent, transparent 6px, rgba(244,63,94,0.12) 6px, rgba(244,63,94,0.12) 7px)",
                                                    }}
                                                />

                                                {/* Left Resize Handle */}
                                                <div
                                                    className="absolute left-0 top-0 bottom-0 w-3 cursor-ew-resize z-20 flex items-center justify-center"
                                                    onPointerDown={(e) => {
                                                        e.stopPropagation();
                                                        hasMovedDuringDragRef.current = false;
                                                        setActiveTrimId(widget.id);
                                                        setActiveSpeedId(null);
                                                        setDragState({
                                                            type: "resize-start",
                                                            widgetKind: "trim",
                                                            widgetId: widget.id,
                                                            startX: e.clientX,
                                                            initialStartTime: widget.startTime,
                                                            initialDuration: widget.duration,
                                                        });
                                                    }}
                                                >
                                                    <div className={`w-1 h-6 rounded-full transition-colors duration-200 ${isActive ? "bg-rose-400" : "bg-rose-400/60 group-hover:bg-rose-400"}`} />
                                                </div>

                                                {/* Label */}
                                                <div className="flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-semibold tracking-wide uppercase pointer-events-none select-none"
                                                    style={{
                                                        background: "rgba(0,0,0,0.55)",
                                                        backdropFilter: "blur(8px)",
                                                        color: "rgb(251,113,133)",
                                                        border: "1px solid rgba(244,63,94,0.3)",
                                                        boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
                                                    }}
                                                >
                                                    <Scissors className="h-3 w-3" />
                                                    <span>Cut {trimWidgets.findIndex((w) => w.id === widget.id) + 1}</span>
                                                </div>

                                                {/* Right Resize Handle */}
                                                <div
                                                    className="absolute right-0 top-0 bottom-0 w-3 cursor-ew-resize z-20 flex items-center justify-center"
                                                    onPointerDown={(e) => {
                                                        e.stopPropagation();
                                                        hasMovedDuringDragRef.current = false;
                                                        setActiveTrimId(widget.id);
                                                        setActiveSpeedId(null);
                                                        setDragState({
                                                            type: "resize-end",
                                                            widgetKind: "trim",
                                                            widgetId: widget.id,
                                                            startX: e.clientX,
                                                            initialStartTime: widget.startTime,
                                                            initialDuration: widget.duration,
                                                        });
                                                    }}
                                                >
                                                    <div className={`w-1 h-6 rounded-full transition-colors duration-200 ${isActive ? "bg-rose-400" : "bg-rose-400/60 group-hover:bg-rose-400"}`} />
                                                </div>
                                            </div>
                                        );
                                    })}

                                {/* Speed Widget Overlays */}
                                {hasVideo &&
                                    speedWidgets.map((widget) => {
                                        const isActive = widget.id === activeSpeedId;
                                        return (
                                            <div
                                                key={widget.id}
                                                tabIndex={0}
                                                role="slider"
                                                aria-label={`Speed segment, ${widget.speed}x, ${widget.startTime.toFixed(1)}s to ${(widget.startTime + widget.duration).toFixed(1)}s. Arrow keys move, shift+arrow resizes.`}
                                                aria-valuenow={widget.startTime}
                                                onFocus={() => {
                                                    setActiveSpeedId(widget.id);
                                                    setActiveTrimId(null);
                                                }}
                                                className={`absolute inset-y-0 z-10 flex items-center justify-center rounded-lg cursor-move group transition-shadow duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-violet-400 ${isActive ? "shadow-[0_0_0_1px_rgba(139,92,246,0.6),0_0_12px_-2px_rgba(139,92,246,0.3)]" : "shadow-[0_0_0_1px_rgba(139,92,246,0.35)]"}`}
                                                style={{
                                                    left: `${widget.startTime * pxPerSecond}px`,
                                                    width: `${widget.duration * pxPerSecond}px`,
                                                    background: isActive
                                                        ? "linear-gradient(135deg, rgba(139,92,246,0.24) 0%, rgba(167,139,250,0.16) 40%, rgba(96,165,250,0.18) 100%)"
                                                        : "linear-gradient(135deg, rgba(139,92,246,0.14) 0%, rgba(167,139,250,0.08) 40%, rgba(96,165,250,0.10) 100%)",
                                                    backdropFilter: "blur(6px)",
                                                }}
                                                onPointerDown={(e) => {
                                                    e.stopPropagation();
                                                    hasMovedDuringDragRef.current = false;
                                                    setActiveSpeedId(widget.id);
                                                    setActiveTrimId(null);
                                                    setDragState({
                                                        type: "move",
                                                        widgetKind: "speed",
                                                        widgetId: widget.id,
                                                        startX: e.clientX,
                                                        initialStartTime: widget.startTime,
                                                        initialDuration: widget.duration,
                                                    });
                                                }}
                                            >
                                                {/* Left Resize Handle */}
                                                <div
                                                    className="absolute left-0 top-0 bottom-0 w-3 cursor-ew-resize z-20 flex items-center justify-center"
                                                    onPointerDown={(e) => {
                                                        e.stopPropagation();
                                                        hasMovedDuringDragRef.current = false;
                                                        setActiveSpeedId(widget.id);
                                                        setActiveTrimId(null);
                                                        setDragState({
                                                            type: "resize-start",
                                                            widgetKind: "speed",
                                                            widgetId: widget.id,
                                                            startX: e.clientX,
                                                            initialStartTime: widget.startTime,
                                                            initialDuration: widget.duration,
                                                        });
                                                    }}
                                                >
                                                    <div className={`w-1 h-6 rounded-full transition-colors duration-200 ${isActive ? "bg-violet-400" : "bg-violet-400/60 group-hover:bg-violet-400"}`} />
                                                </div>

                                                {/* Label */}
                                                <div className="flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-semibold tracking-wide uppercase pointer-events-none select-none"
                                                    style={{
                                                        background: "rgba(0,0,0,0.55)",
                                                        backdropFilter: "blur(8px)",
                                                        color: "rgb(167,139,250)",
                                                        border: "1px solid rgba(139,92,246,0.3)",
                                                        boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
                                                    }}
                                                >
                                                    <FastForward className="h-3 w-3" />
                                                    <span>{widget.speed}×</span>
                                                </div>

                                                {/* Right Resize Handle */}
                                                <div
                                                    className="absolute right-0 top-0 bottom-0 w-3 cursor-ew-resize z-20 flex items-center justify-center"
                                                    onPointerDown={(e) => {
                                                        e.stopPropagation();
                                                        hasMovedDuringDragRef.current = false;
                                                        setActiveSpeedId(widget.id);
                                                        setActiveTrimId(null);
                                                        setDragState({
                                                            type: "resize-end",
                                                            widgetKind: "speed",
                                                            widgetId: widget.id,
                                                            startX: e.clientX,
                                                            initialStartTime: widget.startTime,
                                                            initialDuration: widget.duration,
                                                        });
                                                    }}
                                                >
                                                    <div className={`w-1 h-6 rounded-full transition-colors duration-200 ${isActive ? "bg-violet-400" : "bg-violet-400/60 group-hover:bg-violet-400"}`} />
                                                </div>
                                            </div>
                                        );
                                    })}

                                {/* Playhead - Centered and visible */}
                                {hasVideo && (
                                    <div
                                        onPointerDown={(e) => {
                                            e.stopPropagation();
                                            setIsDraggingPlayhead(true);
                                        }}
                                        className="absolute bottom-0 top-[-10px] z-20 flex flex-col items-center cursor-grab active:cursor-grabbing -translate-x-1/2 pointer-events-auto"
                                        style={{ left: `${currentTime * pxPerSecond}px` }}
                                    >
                                        {/* Head triangle */}
                                        <div
                                            className="h-3 w-3 relative z-30"
                                            style={{
                                                clipPath: "polygon(0 0, 100% 0, 50% 100%)",
                                                background: "hsl(var(--primary))",
                                                filter: "drop-shadow(0 1px 1px rgb(0 0 0 / 0.3))"
                                            }}
                                        />
                                        {/* Line */}
                                        <div className="w-[1.5px] flex-1 bg-primary shadow-[0_0_4px_rgba(0,0,0,0.3)]" />
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </TooltipProvider>
    );
}
