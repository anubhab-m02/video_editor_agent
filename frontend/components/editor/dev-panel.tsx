"use client";

import { X } from "lucide-react";

type TokenEstimateResponse = {
    duration_sec: number;
    direct_video_tokens_est: number;
    sprite_tokens_est: number;
    total_frames: number;
    sheet_count: number;
    recommendation: string;
    notes: string[];
};

type EscalationEvent = {
    window_start_sec: number;
    window_end_sec: number;
    trigger: "user_cue" | "low_confidence";
    confidence_before: number;
    tokens_used: number | null;
};

type DevPanelProps = {
    isOpen: boolean;
    onClose: () => void;
    tokenEstimate: TokenEstimateResponse | null;
    isEstimating: boolean;
    onEstimate: () => void;
    canEstimate: boolean;
    escalationEvents: EscalationEvent[];
    sessionTokensUsed: number;
    escalationThreshold: number;
    onEscalationThresholdChange: (value: number) => void;
};

// Hidden by default (Design Handoff Part 3): internal tool for tuning cost levers
// (sprite interval/thumb_width via the token estimate, escalation threshold) and
// watching real escalation events (ADR-0002/P3-3), not a creator-facing feature.
export function DevPanel({
    isOpen,
    onClose,
    tokenEstimate,
    isEstimating,
    onEstimate,
    canEstimate,
    escalationEvents,
    sessionTokensUsed,
    escalationThreshold,
    onEscalationThresholdChange,
}: DevPanelProps) {
    return (
        <div
            className={`fixed inset-y-0 right-0 z-50 w-80 transform border-l border-white/10 bg-zinc-950/95 p-4 backdrop-blur-xl transition-transform duration-200 ease ${
                isOpen ? "translate-x-0" : "translate-x-full"
            }`}
            style={{ boxShadow: "-8px 0 24px rgba(0,0,0,0.4)" }}
            aria-hidden={!isOpen}
        >
            <div className="flex items-center justify-between">
                <h2 className="text-xs font-semibold uppercase tracking-wide text-amber-300">Dev Panel</h2>
                <button
                    onClick={onClose}
                    aria-label="Close dev panel"
                    className="rounded p-1 text-zinc-400 hover:bg-white/[0.06]"
                >
                    <X className="h-4 w-4" />
                </button>
            </div>

            <div className="mt-4 space-y-4 text-xs">
                <section>
                    <h3 className="font-semibold text-zinc-300">Token estimate</h3>
                    <button
                        onClick={onEstimate}
                        disabled={!canEstimate || isEstimating}
                        className="mt-2 w-full rounded-lg border border-white/[0.08] bg-white/[0.04] py-1.5 text-[11px] font-medium text-zinc-300 disabled:opacity-40"
                    >
                        {isEstimating ? "Estimating..." : "Run estimate"}
                    </button>
                    {tokenEstimate ? (
                        <div className="mt-2 space-y-1.5">
                            <div className="grid grid-cols-2 gap-2">
                                <div className="rounded border border-white/[0.06] bg-white/[0.03] p-2">
                                    <p className="text-zinc-400">Direct Upload</p>
                                    <p className="font-mono text-zinc-100">
                                        {tokenEstimate.direct_video_tokens_est.toLocaleString()}
                                    </p>
                                </div>
                                <div className="rounded border border-white/[0.06] bg-white/[0.03] p-2">
                                    <p className="text-zinc-400">Sprite Sheets</p>
                                    <p className="font-mono text-zinc-100">
                                        {tokenEstimate.sprite_tokens_est.toLocaleString()}
                                    </p>
                                </div>
                            </div>
                            <p className="text-[11px] text-zinc-400">
                                Frames: {tokenEstimate.total_frames} | Sheets: {tokenEstimate.sheet_count}
                            </p>
                            <p className="text-[11px] text-emerald-300">{tokenEstimate.recommendation}</p>
                        </div>
                    ) : (
                        <p className="mt-2 text-[11px] text-zinc-500">No estimate run yet.</p>
                    )}
                </section>

                <section>
                    <h3 className="font-semibold text-zinc-300">Escalation threshold</h3>
                    <div className="mt-2 flex items-center gap-2">
                        <input
                            type="range"
                            min={0}
                            max={1}
                            step={0.05}
                            value={escalationThreshold}
                            onChange={(e) => onEscalationThresholdChange(Number(e.target.value))}
                            className="w-full accent-amber-400"
                            aria-label="Escalation confidence threshold"
                        />
                        <span className="font-mono text-zinc-100">{escalationThreshold.toFixed(2)}</span>
                    </div>
                    <p className="mt-1 text-[11px] text-zinc-500">
                        Applies to the next plan call this session only, not persisted.
                    </p>
                </section>

                <section>
                    <h3 className="font-semibold text-zinc-300">Running totals</h3>
                    <p className="mt-2 font-mono text-zinc-100">
                        {sessionTokensUsed.toLocaleString()} tokens this session
                    </p>
                </section>

                <section>
                    <h3 className="font-semibold text-zinc-300">Escalation events</h3>
                    {escalationEvents.length === 0 ? (
                        <p className="mt-2 text-[11px] text-zinc-500">No escalations this session.</p>
                    ) : (
                        <div className="mt-2 space-y-1.5">
                            {escalationEvents.map((event, i) => (
                                <div
                                    key={i}
                                    className="rounded border border-white/[0.06] bg-white/[0.03] p-2 text-[11px]"
                                >
                                    <div className="flex items-center justify-between">
                                        <span className="font-mono text-zinc-100">
                                            {event.window_start_sec.toFixed(1)}s &rarr; {event.window_end_sec.toFixed(1)}s
                                        </span>
                                        <span className="rounded border border-white/[0.08] bg-white/[0.05] px-1 py-px uppercase tracking-wide text-zinc-400">
                                            {event.trigger === "user_cue" ? "user cue" : "low confidence"}
                                        </span>
                                    </div>
                                    <p className="mt-1 text-zinc-400">
                                        Confidence before: {event.confidence_before.toFixed(2)}
                                        {event.tokens_used != null ? ` · ${event.tokens_used.toLocaleString()} tokens` : ""}
                                    </p>
                                </div>
                            ))}
                        </div>
                    )}
                </section>
            </div>
        </div>
    );
}
