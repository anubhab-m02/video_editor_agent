"use client";

import { useState, useRef, useEffect } from "react";
import { Download, Sparkles, Send, Bot, User, Film, Check, X, Undo2 } from "lucide-react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { useVideo } from "./video-context";
import { fingerprintsMatch, type FileFingerprint } from "./video-store";
import { DevPanel } from "./dev-panel";

type ChatMessage = {
    id: string;
    role: "user" | "assistant";
    content: string;
};

type ChatTurn = {
    role: "user" | "assistant";
    content: string;
};

type SpriteSheet = {
    sheet_index: number;
    image_url: string;
    image_width: number;
    image_height: number;
    start_time_sec: number;
    end_time_sec: number;
};

type SpriteAnalysisResponse = {
    duration_sec: number;
    interval_sec: number;
    total_frames: number;
    sprite_job_id: string;
    sheets: SpriteSheet[];
};

type TokenEstimateResponse = {
    duration_sec: number;
    direct_video_tokens_est: number;
    sprite_tokens_est: number;
    total_frames: number;
    sheet_count: number;
    recommendation: string;
    notes: string[];
};

type CutSuggestion = {
    id: string;
    action: "trim_video" | "speed_video";
    operation: "remove_segment" | "extract_range" | "apply_speed_range";
    start_sec: number;
    end_sec: number;
    reason: string;
    confidence: number;
    speed_multiplier?: number | null;
};

type EscalationEvent = {
    window_start_sec: number;
    window_end_sec: number;
    trigger: "user_cue" | "low_confidence";
    confidence_before: number;
    tokens_used: number | null;
};

type AgentPlanResponse = {
    plan_id: string;
    reasoning: string;
    proposals: CutSuggestion[];
    model: string;
    strategy: string;
    tokens_used: number | null;
    escalation: EscalationEvent | null;
};

type ProposalStatus = "pending" | "accepted" | "rejected";
type Proposal = CutSuggestion & { status: ProposalStatus };

type ExportResponse = {
    output_url: string;
    output_name: string;
    removed_ranges_count: number;
};

const PLACEHOLDER_MESSAGES: ChatMessage[] = [
    {
        id: "1",
        role: "assistant",
        content:
            "Hey! I can trim, cut, extract, speed up sections, and detect dead air/silence. Upload a video and iterate with me.",
    },
];
const MAX_VIDEO_DURATION_SEC = 1200; // 20 min, ADR-0006
const MAX_CONTEXT_MESSAGES = 10;
const CHAT_STORAGE_KEY = "video-editor-chat-session";

type StoredChatSession = { fingerprint: FileFingerprint; messages: ChatMessage[] };

function loadStoredChat(): StoredChatSession | null {
    if (typeof window === "undefined") return null;
    try {
        const raw = window.localStorage.getItem(CHAT_STORAGE_KEY);
        return raw ? (JSON.parse(raw) as StoredChatSession) : null;
    } catch {
        return null;
    }
}
const RANGE_EPSILON_SEC = 0.03;

function formatDurationLabel(totalSeconds: number): string {
    const minutes = Math.round(totalSeconds / 60);
    return minutes >= 1 ? `${minutes} min` : `${totalSeconds}s`;
}

// ponytail: fixed frame budget keeps sprite sheet count (and FFmpeg calls) bounded
// regardless of video length, instead of the old hardcoded 0.25s interval blowing up
// at the new 20-min ceiling. Clamped to 0.25s minimum to match prior short-clip behavior.
const TARGET_SPRITE_FRAMES = 48;
function computeSpriteInterval(durationSec: number): number {
    if (durationSec <= 0) return 0.25;
    return Math.max(0.25, durationSec / TARGET_SPRITE_FRAMES);
}

// P3-5: turns that have fallen out of buildChatHistory's raw sliding window —
// the candidate pool for summarization. Pure so it's cheap to call every send.
function getOlderTurns(messages: ChatMessage[]): ChatTurn[] {
    const turns = messages.filter((m) => m.content.trim().length > 0);
    if (turns.length <= MAX_CONTEXT_MESSAGES) return [];
    return turns.slice(0, -MAX_CONTEXT_MESSAGES).map((m) => ({
        role: m.role,
        content: m.content.replace(/\s+/g, " ").trim().slice(0, 300),
    }));
}

function buildChatHistory(messages: ChatMessage[]): ChatTurn[] {
    return messages
        .slice(-MAX_CONTEXT_MESSAGES)
        .map((m) => ({
            role: m.role,
            content: m.content.replace(/\s+/g, " ").trim().slice(0, 300),
        }))
        .filter((m) => m.content.length > 0);
}

function mergeTrimSuggestions(
    existing: Array<{ start: number; end: number }>,
    incoming: CutSuggestion[]
): Array<{ start: number; end: number }> {
    const next = [...existing];
    for (const item of incoming) {
        const duplicate = next.some(
            (r) =>
                Math.abs(r.start - item.start_sec) <= RANGE_EPSILON_SEC &&
                Math.abs(r.end - item.end_sec) <= RANGE_EPSILON_SEC
        );
        if (!duplicate) {
            next.push({ start: item.start_sec, end: item.end_sec });
        }
    }
    return next;
}

function mergeSpeedSuggestions(
    existing: Array<{ start: number; end: number; speed: number }>,
    incoming: CutSuggestion[]
): Array<{ start: number; end: number; speed: number }> {
    const next = [...existing];
    for (const item of incoming) {
        const speed = item.speed_multiplier && item.speed_multiplier > 0 ? item.speed_multiplier : 2;
        const duplicate = next.some(
            (r) =>
                Math.abs(r.start - item.start_sec) <= RANGE_EPSILON_SEC &&
                Math.abs(r.end - item.end_sec) <= RANGE_EPSILON_SEC &&
                Math.abs(r.speed - speed) <= 0.01
        );
        if (!duplicate) {
            next.push({ start: item.start_sec, end: item.end_sec, speed });
        }
    }
    return next;
}

type TrimLike = { start: number; end: number };
type SpeedLike = { start: number; end: number; speed: number };
type ProposalState = { proposals: Proposal[]; trimRanges: TrimLike[]; speedRanges: SpeedLike[] };

// Pure state-transition functions for the preview-before-apply flow (PRD P0-2).
// Kept free of React state so the accept/reject/undo logic is directly testable.
export function applyProposalAccept(
    proposals: Proposal[],
    trimRanges: TrimLike[],
    speedRanges: SpeedLike[],
    id: string
): ProposalState {
    const proposal = proposals.find((p) => p.id === id);
    if (!proposal || proposal.status !== "pending") {
        return { proposals, trimRanges, speedRanges };
    }
    return {
        proposals: proposals.map((p) => (p.id === id ? { ...p, status: "accepted" } : p)),
        trimRanges: proposal.action === "trim_video" ? mergeTrimSuggestions(trimRanges, [proposal]) : trimRanges,
        speedRanges: proposal.action === "speed_video" ? mergeSpeedSuggestions(speedRanges, [proposal]) : speedRanges,
    };
}

export function applyProposalReject(proposals: Proposal[], id: string): Proposal[] {
    return proposals.map((p) => (p.id === id ? { ...p, status: "rejected" } : p));
}

export function applyProposalUndo(
    proposals: Proposal[],
    trimRanges: TrimLike[],
    speedRanges: SpeedLike[],
    id: string
): ProposalState {
    const proposal = proposals.find((p) => p.id === id);
    if (!proposal || proposal.status !== "accepted") {
        return { proposals, trimRanges, speedRanges };
    }
    let nextTrim = trimRanges;
    let nextSpeed = speedRanges;
    if (proposal.action === "trim_video") {
        nextTrim = trimRanges.filter(
            (r) =>
                !(
                    Math.abs(r.start - proposal.start_sec) <= RANGE_EPSILON_SEC &&
                    Math.abs(r.end - proposal.end_sec) <= RANGE_EPSILON_SEC
                )
        );
    } else {
        const speed = proposal.speed_multiplier && proposal.speed_multiplier > 0 ? proposal.speed_multiplier : 2;
        nextSpeed = speedRanges.filter(
            (r) =>
                !(
                    Math.abs(r.start - proposal.start_sec) <= RANGE_EPSILON_SEC &&
                    Math.abs(r.end - proposal.end_sec) <= RANGE_EPSILON_SEC &&
                    Math.abs(r.speed - speed) <= 0.01
                )
        );
    }
    return {
        proposals: proposals.map((p) => (p.id === id ? { ...p, status: "pending" } : p)),
        trimRanges: nextTrim,
        speedRanges: nextSpeed,
    };
}

export function applyProposalAcceptAll(
    proposals: Proposal[],
    trimRanges: TrimLike[],
    speedRanges: SpeedLike[]
): ProposalState {
    const pending = proposals.filter((p) => p.status === "pending");
    if (pending.length === 0) return { proposals, trimRanges, speedRanges };
    const trimItems = pending.filter((p) => p.action === "trim_video");
    const speedItems = pending.filter((p) => p.action === "speed_video");
    return {
        proposals: proposals.map((p) => (p.status === "pending" ? { ...p, status: "accepted" } : p)),
        trimRanges: trimItems.length > 0 ? mergeTrimSuggestions(trimRanges, trimItems) : trimRanges,
        speedRanges: speedItems.length > 0 ? mergeSpeedSuggestions(speedRanges, speedItems) : speedRanges,
    };
}

export function applyProposalRejectAll(proposals: Proposal[]): Proposal[] {
    return proposals.map((p) => (p.status === "pending" ? { ...p, status: "rejected" } : p));
}

// Each /agent/plan call is a full fresh re-analysis, not an addendum to the last
// one — it supersedes whatever from the previous turn is still pending
// (unactioned). Already accepted/rejected proposals are the user's real
// decisions and are kept regardless.
export function mergeNewProposals(existing: Proposal[], incoming: Proposal[]): Proposal[] {
    return [...existing.filter((p) => p.status !== "pending"), ...incoming];
}

export function Inspector() {
    const { sourceFile, duration, trimRanges, speedRanges, setTrimRanges, setSpeedRanges } = useVideo();
    const [messages, setMessages] = useState<ChatMessage[]>(PLACEHOLDER_MESSAGES);
    const [input, setInput] = useState("");
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [isEstimating, setIsEstimating] = useState(false);
    const [spriteData, setSpriteData] = useState<SpriteAnalysisResponse | null>(null);
    const [tokenEstimate, setTokenEstimate] = useState<TokenEstimateResponse | null>(null);
    const [proposals, setProposals] = useState<Proposal[]>([]);
    const [isSuggesting, setIsSuggesting] = useState(false);
    const [isExporting, setIsExporting] = useState(false);
    const [exportResult, setExportResult] = useState<ExportResponse | null>(null);
    // P3-5: rolling real-model summary of chat turns that have fallen out of the
    // raw context window (buildChatHistory's slice), replacing the old
    // last-3-messages heuristic. summarizedThroughCount tracks how many "older"
    // messages are already folded in, so only newly-overflowed turns get sent
    // to /agent/summarize — cost stays flat regardless of session length.
    const [conversationSummary, setConversationSummary] = useState("");
    const [summarizedThroughCount, setSummarizedThroughCount] = useState(0);
    // Dev-panel-only (Design Handoff Part 3): session log of real escalation events
    // and a live-adjustable threshold override, not persisted, not creator-facing.
    const [escalationEvents, setEscalationEvents] = useState<EscalationEvent[]>([]);
    const [sessionTokensUsed, setSessionTokensUsed] = useState(0);
    const [escalationThreshold, setEscalationThreshold] = useState(0.6);
    const [isDevPanelOpen, setIsDevPanelOpen] = useState(
        () => process.env.NEXT_PUBLIC_DEV_PANEL === "true"
    );
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const isVideoTooLong = duration > MAX_VIDEO_DURATION_SEC;
    const durationLabel =
        Number.isFinite(duration) && duration > 0 ? `${duration.toFixed(2)}s` : "unknown";

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    // Dev/debug panel toggle (Design Handoff Part 3) — hidden by default, never a
    // creator-facing feature.
    useEffect(() => {
        function onKeyDown(e: KeyboardEvent) {
            if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === "d") {
                e.preventDefault();
                setIsDevPanelOpen((prev) => !prev);
            }
        }
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, []);

    // Auto-resize textarea to fit content
    useEffect(() => {
        const el = textareaRef.current;
        if (!el) return;
        el.style.height = "auto";
        el.style.height = `${el.scrollHeight}px`;
    }, [input]);

    // Session persistence (PRD P1-2): restore chat only when the same file is
    // re-loaded (e.g. after a refresh); a different file always starts fresh,
    // matching the store's trimRanges/speedRanges restore behavior.
    useEffect(() => {
        if (!sourceFile) return;
        const fingerprint: FileFingerprint = {
            name: sourceFile.name,
            size: sourceFile.size,
            lastModified: sourceFile.lastModified,
        };
        const stored = loadStoredChat();
        setMessages(
            stored && fingerprintsMatch(stored.fingerprint, fingerprint) ? stored.messages : PLACEHOLDER_MESSAGES
        );
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [sourceFile]);

    useEffect(() => {
        if (!sourceFile || typeof window === "undefined") return;
        const fingerprint: FileFingerprint = {
            name: sourceFile.name,
            size: sourceFile.size,
            lastModified: sourceFile.lastModified,
        };
        try {
            window.localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify({ fingerprint, messages }));
        } catch {
            // Best-effort — localStorage may be unavailable or full.
        }
    }, [messages, sourceFile]);

    // P3-5: rolling real-model summary. Only calls /agent/summarize when the
    // "older" bucket has actually grown since last time — folds just the newly
    // overflowed turns into the cached summary instead of re-summarizing the
    // whole history every send. Falls back to whatever's cached on failure.
    async function ensureConversationSummary(nextMessages: ChatMessage[]): Promise<string> {
        const older = getOlderTurns(nextMessages);
        if (older.length <= summarizedThroughCount) return conversationSummary;
        const newTurns = older.slice(summarizedThroughCount);
        try {
            const response = await fetch("/api/agent/summarize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    older_turns: newTurns,
                    previous_summary: conversationSummary || undefined,
                }),
            });
            if (!response.ok) return conversationSummary;
            const data = (await response.json()) as { summary: string };
            setConversationSummary(data.summary);
            setSummarizedThroughCount(older.length);
            return data.summary;
        } catch {
            return conversationSummary;
        }
    }

    async function handleSend() {
        if (!input.trim()) return;
        if (isVideoTooLong) {
            setMessages((prev) => [
                ...prev,
                {
                    id: Date.now().toString(),
                    role: "assistant",
                    content: `AI tools are limited to videos up to ${formatDurationLabel(MAX_VIDEO_DURATION_SEC)}. Current video: ${durationLabel}.`,
                },
            ]);
            return;
        }
        const userMsg: ChatMessage = {
            id: Date.now().toString(),
            role: "user",
            content: input.trim(),
        };
        const nextMessages = [...messages, userMsg];
        setMessages((prev) => [...prev, userMsg]);
        setInput("");
        setIsSuggesting(true);
        // ADR-0007/direct-video: every plan call needs the persisted upload +
        // sprite_job_id regardless of strategy — establish it transparently on
        // first send instead of making the user click "Sprites" manually first.
        let activeSpriteData = spriteData;
        if (!activeSpriteData) {
            activeSpriteData = await uploadForAnalysis(false);
            if (!activeSpriteData) {
                setIsSuggesting(false);
                return;
            }
        }

        const conversationSummaryForCall = await ensureConversationSummary(nextMessages);

        try {
            const response = await fetch("/api/agent/plan", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    prompt: userMsg.content,
                    duration_sec: activeSpriteData.duration_sec,
                    sprite_interval_sec: activeSpriteData.interval_sec,
                    total_frames: activeSpriteData.total_frames,
                    sheets_count: activeSpriteData.sheets.length,
                    sprite_job_id: activeSpriteData.sprite_job_id,
                    chat_history: buildChatHistory(nextMessages),
                    conversation_summary: conversationSummaryForCall,
                    trim_ranges: trimRanges,
                    speed_ranges: speedRanges,
                    escalation_confidence_threshold: escalationThreshold,
                }),
            });
            const data = (await response.json()) as AgentPlanResponse | { detail?: string };
            if (!response.ok) {
                const message = "detail" in data ? data.detail : undefined;
                throw new Error(message || "Failed to plan edits.");
            }
            const result = data as AgentPlanResponse;
            // Preview-before-apply: the plan's proposals arrive pending only.
            // Nothing touches the timeline until the user accepts (PRD P0-2).
            const nextProposals: Proposal[] = result.proposals.map((s) => ({
                ...s,
                status: "pending",
            }));
            setProposals((prev) => mergeNewProposals(prev, nextProposals));
            if (result.tokens_used) {
                setSessionTokensUsed((prev) => prev + (result.tokens_used ?? 0));
            }
            if (result.escalation) {
                setEscalationEvents((prev) => [result.escalation as EscalationEvent, ...prev]);
            }
            setMessages((prev) => [
                ...prev,
                {
                    id: (Date.now() + 1).toString(),
                    role: "assistant",
                    content: `${result.reasoning} (${result.model}, ${result.strategy})`,
                },
            ]);
        } catch (error) {
            setMessages((prev) => [
                ...prev,
                {
                    id: (Date.now() + 1).toString(),
                    role: "assistant",
                    content:
                        error instanceof Error
                            ? error.message
                            : "Failed to suggest edits.",
                },
            ]);
        } finally {
            setIsSuggesting(false);
        }
    }

    function acceptProposal(id: string) {
        const result = applyProposalAccept(proposals, trimRanges, speedRanges, id);
        setProposals(result.proposals);
        setTrimRanges(result.trimRanges);
        setSpeedRanges(result.speedRanges);
    }

    function rejectProposal(id: string) {
        setProposals(applyProposalReject(proposals, id));
    }

    function undoProposal(id: string) {
        const result = applyProposalUndo(proposals, trimRanges, speedRanges, id);
        setProposals(result.proposals);
        setTrimRanges(result.trimRanges);
        setSpeedRanges(result.speedRanges);
    }

    function acceptAllPending() {
        const result = applyProposalAcceptAll(proposals, trimRanges, speedRanges);
        setProposals(result.proposals);
        setTrimRanges(result.trimRanges);
        setSpeedRanges(result.speedRanges);
    }

    function rejectAllPending() {
        setProposals(applyProposalRejectAll(proposals));
    }

    function handleKeyDown(e: React.KeyboardEvent) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    }

    // Establishes the persisted upload + sprite_job_id every /agent/plan call
    // needs (ADR-0007) — regardless of whether the direct-video path (short
    // clips) ends up using the sprite thumbnails at all. Shared by the manual
    // "Sprites" button (announceResult=true, shows thumbnails + a chat message)
    // and the AI chat's silent auto-upload on first send (announceResult=false).
    async function uploadForAnalysis(announceResult: boolean): Promise<SpriteAnalysisResponse | null> {
        if (!sourceFile) return null;
        setIsAnalyzing(true);
        setSpriteData(null);

        const form = new FormData();
        form.append("file", sourceFile);
        form.append("interval_sec", computeSpriteInterval(duration).toFixed(3));
        form.append("columns", "8");
        form.append("rows", "8");
        form.append("thumb_width", "256");

        try {
            const response = await fetch("/api/analyze/sprites", {
                method: "POST",
                body: form,
            });
            const data = (await response.json()) as SpriteAnalysisResponse | { detail?: string };
            if (!response.ok) {
                const message =
                    "detail" in data ? data.detail : undefined;
                throw new Error(message || "Failed to generate sprite sheets.");
            }
            const spriteResponse = data as SpriteAnalysisResponse;
            setSpriteData(spriteResponse);
            if (announceResult) {
                setMessages((prev) => [
                    ...prev,
                    {
                        id: Date.now().toString(),
                        role: "assistant",
                        content: `Generated ${spriteResponse.sheets.length} sprite sheet(s) for ${spriteResponse.duration_sec}s video.`,
                    },
                ]);
            }
            return spriteResponse;
        } catch (error) {
            setMessages((prev) => [
                ...prev,
                {
                    id: Date.now().toString(),
                    role: "assistant",
                    content:
                        error instanceof Error
                            ? error.message
                            : "Failed to generate sprite sheets.",
                },
            ]);
            return null;
        } finally {
            setIsAnalyzing(false);
        }
    }

    async function handleGenerateSprites() {
        if (!sourceFile || isAnalyzing || isVideoTooLong) return;
        await uploadForAnalysis(true);
    }

    async function handleEstimateTokens() {
        if (!sourceFile || isEstimating || isVideoTooLong) return;
        setIsEstimating(true);

        const form = new FormData();
        form.append("file", sourceFile);
        form.append("interval_sec", computeSpriteInterval(duration).toFixed(3));
        form.append("columns", "8");
        form.append("rows", "8");
        form.append("thumb_width", "256");

        try {
            const response = await fetch("/api/analyze/token-estimate-from-file", {
                method: "POST",
                body: form,
            });
            const data = (await response.json()) as TokenEstimateResponse | { detail?: string };
            if (!response.ok) {
                const message = "detail" in data ? data.detail : undefined;
                throw new Error(message || "Failed to estimate tokens.");
            }
            const estimate = data as TokenEstimateResponse;
            setTokenEstimate(estimate);
        } catch (error) {
            setMessages((prev) => [
                ...prev,
                {
                    id: Date.now().toString(),
                    role: "assistant",
                    content:
                        error instanceof Error
                            ? error.message
                            : "Failed to estimate token usage.",
                },
            ]);
        } finally {
            setIsEstimating(false);
        }
    }

    async function handleExportVideo() {
        if (!sourceFile || isExporting || isVideoTooLong) return;
        setIsExporting(true);
        setExportResult(null);

        function buildExportForm(useSourceFile: boolean): FormData {
            const form = new FormData();
            if (useSourceFile) {
                form.append("file", sourceFile as File);
            } else {
                form.append("sprite_job_id", spriteData!.sprite_job_id);
            }
            form.append("trim_ranges", JSON.stringify(trimRanges));
            form.append("speed_ranges", JSON.stringify(speedRanges));
            if (speedRanges.length === 0) {
                form.append("speed_multiplier", "1");
                form.append("speed", "1x");
            }
            return form;
        }

        // ADR-0007: reference the source already persisted under sprite_job_id
        // instead of re-uploading the whole file, when one exists. Falls back to
        // a direct upload if the persisted source expired/was swept (404).
        const hasPersistedSource = Boolean(spriteData?.sprite_job_id);

        try {
            let response = await fetch("/api/export/from-file", {
                method: "POST",
                body: buildExportForm(!hasPersistedSource),
            });
            if (response.status === 404 && hasPersistedSource) {
                response = await fetch("/api/export/from-file", {
                    method: "POST",
                    body: buildExportForm(true),
                });
            }
            const data = (await response.json()) as ExportResponse | { detail?: string };
            if (!response.ok) {
                const message = "detail" in data ? data.detail : undefined;
                throw new Error(message || "Failed to export video.");
            }
            const result = data as ExportResponse;
            setExportResult(result);

            const downloadResponse = await fetch(result.output_url);
            if (!downloadResponse.ok) {
                throw new Error("Export succeeded but download failed.");
            }
            const blob = await downloadResponse.blob();
            const blobUrl = URL.createObjectURL(blob);
            const anchor = document.createElement("a");
            anchor.href = blobUrl;
            anchor.download = result.output_name || "export.mp4";
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            URL.revokeObjectURL(blobUrl);

            setMessages((prev) => [
                ...prev,
                {
                    id: Date.now().toString(),
                    role: "assistant",
                    content: `Export complete. Download started. Applied ${result.removed_ranges_count} trim range(s).`,
                },
            ]);
        } catch (error) {
            setMessages((prev) => [
                ...prev,
                {
                    id: Date.now().toString(),
                    role: "assistant",
                    content:
                        error instanceof Error
                            ? error.message
                            : "Export failed.",
                },
            ]);
        } finally {
            setIsExporting(false);
        }
    }

    const pendingProposalCount = proposals.filter((p) => p.status === "pending").length;

    return (
        <>
        <div
            className="flex h-full w-[320px] shrink-0 flex-col"
            style={{
                borderLeft: "1px solid var(--color-border)",
                background: "linear-gradient(180deg, var(--color-surface) 0%, color-mix(in srgb, var(--color-bg) 80%, transparent) 100%)",
                backdropFilter: "blur(20px)",
            }}
        >
            {/* Action dock */}
            <div className="p-3 space-y-2.5">
                {/* Hero export button */}
                <button
                    className="group relative w-full rounded-xl py-2.5 text-sm font-semibold text-white transition-all duration-200 ease disabled:opacity-40 disabled:cursor-not-allowed"
                    style={{
                        background: "linear-gradient(180deg, var(--color-primary) 0%, var(--color-primary-deep) 100%)",
                        boxShadow: "0 0 0 1px rgba(52,211,153,0.3), 0 1px 0 0 rgba(255,255,255,0.1) inset, 0 4px 16px -4px var(--color-primary-glow), 0 1px 2px rgba(0,0,0,0.2)",
                    }}
                    onClick={handleExportVideo}
                    disabled={!sourceFile || isExporting || isVideoTooLong}
                >
                    <span className="flex items-center justify-center gap-2">
                        <Download className="h-4 w-4" />
                        {isExporting ? "Exporting..." : "Export Video"}
                    </span>
                </button>

                {exportResult ? (
                    <a
                        href={exportResult.output_url}
                        target="_blank"
                        rel="noreferrer"
                        className="block rounded-lg px-3 py-2 text-center text-xs text-emerald-300 transition-colors duration-200 ease hover:bg-emerald-500/15"
                        style={{
                            border: "1px solid rgba(52,211,153,0.2)",
                            background: "color-mix(in srgb, var(--color-primary) 6%, transparent)",
                        }}
                    >
                        Open Export: {exportResult.output_name}
                    </a>
                ) : null}

                {/* Secondary actions — compact glass row */}
                <div className="flex gap-2">
                    <button
                        className="flex flex-1 items-center justify-center gap-1.5 rounded-lg py-2 text-xs font-medium text-zinc-300 transition-all duration-200 ease hover:text-zinc-100 disabled:opacity-40 disabled:cursor-not-allowed"
                        style={{
                            border: "1px solid var(--color-border)",
                            background: "var(--color-surface-raised)",
                            boxShadow: "var(--inset-highlight)",
                        }}
                        onClick={handleGenerateSprites}
                        disabled={!sourceFile || isAnalyzing || isVideoTooLong}
                    >
                        <Film className="h-3.5 w-3.5 text-zinc-400" />
                        {isAnalyzing ? "Generating..." : "Sprites"}
                    </button>
                </div>

                {isVideoTooLong ? (
                    <p
                        className="rounded-lg px-2.5 py-1.5 text-xs text-amber-300"
                        style={{
                            border: "1px solid color-mix(in srgb, var(--color-warning) 20%, transparent)",
                            background: "color-mix(in srgb, var(--color-warning) 6%, transparent)",
                        }}
                    >
                        AI/Export is limited to {formatDurationLabel(MAX_VIDEO_DURATION_SEC)} max. Current: {durationLabel}.
                    </p>
                ) : null}
            </div>

            <Separator className="bg-white/[0.06]" />

            {/* Chat header */}
            <div className="flex items-center gap-2 px-4 py-3">
                <Sparkles className="h-4 w-4 text-primary" />
                <h3 className="text-sm font-semibold text-zinc-200">Edit with AI</h3>
            </div>

            <Separator className="bg-white/[0.06]" />

            {/* Chat messages */}
            <ScrollArea className="flex-1 px-3">
                <div className="space-y-3 py-3">
                    {spriteData ? (
                        <div
                            className="rounded-lg p-2"
                            style={{
                                border: "1px solid var(--color-border)",
                                background: "var(--color-surface-raised)",
                                boxShadow: "var(--inset-highlight)",
                            }}
                        >
                            <p className="text-xs text-zinc-400">
                                Sprites: {spriteData.sheets.length} sheets, {spriteData.total_frames} frames
                            </p>
                            {spriteData.sheets.some((sheet) => sheet.image_url) ? (
                                <div className="mt-2 space-y-2">
                                    {spriteData.sheets
                                        .filter((sheet) => Boolean(sheet.image_url))
                                        .slice(0, 2)
                                        .map((sheet) => (
                                            <div key={sheet.sheet_index} className="space-y-1">
                                                <Image
                                                    src={sheet.image_url}
                                                    alt={`Sprite sheet ${sheet.sheet_index}`}
                                                    width={sheet.image_width}
                                                    height={sheet.image_height}
                                                    unoptimized
                                                    className="w-full rounded border border-zinc-700"
                                                />
                                                <p className="text-[11px] text-zinc-400">
                                                    Sheet {sheet.sheet_index}: {sheet.start_time_sec}s to {sheet.end_time_sec}s
                                                </p>
                                            </div>
                                        ))}
                                </div>
                            ) : (
                                <p className="mt-2 text-[11px] text-zinc-400">
                                    Sprite files are temporary and not persisted in this environment.
                                </p>
                            )}
                        </div>
                    ) : null}
                    {messages.map((msg) => (
                        <div
                            key={msg.id}
                            className={`flex min-w-0 gap-2.5 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
                        >
                            <div
                                className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full ${msg.role === "assistant"
                                    ? "bg-primary/20 text-primary"
                                    : "bg-zinc-700 text-zinc-300"
                                    }`}
                            >
                                {msg.role === "assistant" ? (
                                    <Bot className="h-3.5 w-3.5" />
                                ) : (
                                    <User className="h-3.5 w-3.5" />
                                )}
                            </div>
                            <div
                                className={`min-w-0 break-words rounded-lg px-3 py-2 text-[13px] leading-relaxed ${msg.role === "assistant"
                                    ? "text-zinc-300"
                                    : "text-zinc-200"
                                    }`}
                                style={{
                                    background: msg.role === "assistant"
                                        ? "linear-gradient(180deg, rgba(39,39,42,0.6) 0%, rgba(24,24,27,0.7) 100%)"
                                        : "var(--color-primary-tint)",
                                    border: `1px solid ${msg.role === "assistant" ? "var(--color-border)" : "color-mix(in srgb, var(--color-primary) 15%, transparent)"}`,
                                }}
                            >
                                {msg.content}
                            </div>
                        </div>
                    ))}
                    {proposals.length > 0 ? (
                        <div
                            className="rounded-lg p-3"
                            style={{
                                border: "1px solid var(--color-border)",
                                background: "var(--color-surface-raised)",
                                boxShadow: "var(--inset-highlight)",
                            }}
                        >
                            <div className="flex items-center justify-between">
                                <p className="text-xs font-semibold text-zinc-300">
                                    AI Proposals{pendingProposalCount > 0 ? ` (${pendingProposalCount} pending)` : ""}
                                </p>
                                {pendingProposalCount > 0 ? (
                                    <div className="flex gap-1.5">
                                        <button
                                            onClick={acceptAllPending}
                                            className="rounded px-1.5 py-0.5 text-[10px] font-medium text-emerald-300 transition-colors duration-200 ease hover:bg-emerald-500/15"
                                        >
                                            Accept all
                                        </button>
                                        <button
                                            onClick={rejectAllPending}
                                            className="rounded px-1.5 py-0.5 text-[10px] font-medium text-zinc-400 transition-colors duration-200 ease hover:bg-white/[0.06]"
                                        >
                                            Reject all
                                        </button>
                                    </div>
                                ) : null}
                            </div>
                            <div className="mt-2 space-y-1.5">
                                {proposals.map((p) => (
                                    <div
                                        key={p.id}
                                        role="listitem"
                                        aria-label={`${p.action === "speed_video" ? "Speed" : "Trim"} proposal, ${p.start_sec.toFixed(2)} to ${p.end_sec.toFixed(2)} seconds, ${Math.round(p.confidence * 100)}% confidence, ${p.status}`}
                                        className={`rounded border px-2 py-1.5 text-[11px] transition-opacity duration-200 ease ${p.status === "rejected" ? "opacity-40" : ""}`}
                                        style={{
                                            borderColor: p.status === "accepted" ? "rgba(52,211,153,0.35)" : "var(--color-border)",
                                            background: p.status === "accepted" ? "color-mix(in srgb, var(--color-primary) 6%, transparent)" : "rgba(255,255,255,0.03)",
                                        }}
                                    >
                                        <div className="flex items-center gap-2 text-zinc-300">
                                            <span className="rounded border border-white/[0.08] bg-white/[0.05] px-1 py-px text-[10px] uppercase tracking-wide text-zinc-400">
                                                {p.action === "speed_video" ? `Speed ${p.speed_multiplier ?? 2}x` : "Trim"}
                                            </span>
                                            <span className="font-mono text-zinc-100">
                                                {p.start_sec.toFixed(2)}s → {p.end_sec.toFixed(2)}s
                                            </span>
                                            <span className={p.confidence < 0.5 ? "text-amber-300" : "text-zinc-400"}>
                                                ({Math.round(p.confidence * 100)}%)
                                            </span>
                                            <div className="ml-auto flex shrink-0 items-center gap-1">
                                                {p.status === "pending" ? (
                                                    <>
                                                        <button
                                                            onClick={() => acceptProposal(p.id)}
                                                            aria-label="Accept proposal"
                                                            className="rounded p-0.5 text-emerald-300 transition-colors duration-200 ease hover:bg-emerald-500/15"
                                                        >
                                                            <Check className="h-3.5 w-3.5" />
                                                        </button>
                                                        <button
                                                            onClick={() => rejectProposal(p.id)}
                                                            aria-label="Reject proposal"
                                                            className="rounded p-0.5 text-zinc-400 transition-colors duration-200 ease hover:bg-white/[0.06]"
                                                        >
                                                            <X className="h-3.5 w-3.5" />
                                                        </button>
                                                    </>
                                                ) : p.status === "accepted" ? (
                                                    <>
                                                        <span className="text-[10px] text-emerald-300">Applied</span>
                                                        <button
                                                            onClick={() => undoProposal(p.id)}
                                                            aria-label="Undo applied proposal"
                                                            className="rounded p-0.5 text-zinc-400 transition-colors duration-200 ease hover:bg-white/[0.06]"
                                                        >
                                                            <Undo2 className="h-3.5 w-3.5" />
                                                        </button>
                                                    </>
                                                ) : (
                                                    <span className="text-[10px] text-zinc-400">Rejected</span>
                                                )}
                                            </div>
                                        </div>
                                        <p className="mt-0.5 text-zinc-400">{p.reason}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : null}
                    <div ref={messagesEndRef} />
                </div>
            </ScrollArea>

            <Separator className="bg-zinc-800" />

            {/* Chat input */}
            <div className="p-3">
                <div
                    className="flex items-end gap-2 rounded-xl px-3 py-2.5 ring-1 ring-transparent transition-all duration-200 ease focus-within:ring-primary/40"
                    style={{
                        border: "1px solid var(--color-border-strong)",
                        background: "var(--color-surface-raised)",
                        boxShadow: "var(--inset-highlight), inset 0 -1px 2px rgba(0,0,0,0.2)",
                    }}
                >
                    <textarea
                        ref={textareaRef}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Describe your edit..."
                        rows={1}
                        className="scrollbar-hide max-h-28 min-h-[1.5rem] w-full resize-none overflow-y-auto bg-transparent text-[13px] leading-relaxed text-zinc-200 placeholder:text-zinc-500 placeholder:transition-colors placeholder:duration-200 focus:outline-none focus:placeholder:text-zinc-600"
                    />
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={handleSend}
                        disabled={!input.trim() || isSuggesting || isVideoTooLong}
                        className="h-7 w-7 shrink-0 rounded-lg text-primary transition-all duration-200 ease hover:bg-primary/15 hover:text-primary active:scale-95 disabled:text-zinc-600"
                    >
                        <Send className="h-3.5 w-3.5" />
                    </Button>
                </div>
                <div className="mt-2 flex items-center justify-center gap-1.5 text-[11px] text-zinc-400">
                    <span>Try: &quot;Trim from 00:12 to 00:47&quot;</span>
                    <kbd className="rounded border border-white/[0.06] bg-white/[0.03] px-1 py-px font-mono text-[10px] leading-none text-zinc-400">⏎</kbd>
                </div>
            </div>
        </div>
        <DevPanel
            isOpen={isDevPanelOpen}
            onClose={() => setIsDevPanelOpen(false)}
            tokenEstimate={tokenEstimate}
            isEstimating={isEstimating}
            onEstimate={handleEstimateTokens}
            canEstimate={Boolean(sourceFile) && !isVideoTooLong}
            escalationEvents={escalationEvents}
            sessionTokensUsed={sessionTokensUsed}
            escalationThreshold={escalationThreshold}
            onEscalationThresholdChange={setEscalationThreshold}
        />
        </>
    );
}
