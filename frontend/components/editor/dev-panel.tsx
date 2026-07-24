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

type DevPanelProps = {
    isOpen: boolean;
    onClose: () => void;
    tokenEstimate: TokenEstimateResponse | null;
    isEstimating: boolean;
    onEstimate: () => void;
    canEstimate: boolean;
};

// Hidden by default (Design Handoff Part 3): internal tool for tuning cost levers
// (sprite interval/thumb_width via the token estimate), not a creator-facing feature.
// The "escalation events" section this was designed alongside (System Design §5's
// direct-video escalation, X1b) hasn't been built yet — that section is an honest
// placeholder rather than fabricated data for a feature that doesn't exist.
export function DevPanel({ isOpen, onClose, tokenEstimate, isEstimating, onEstimate, canEstimate }: DevPanelProps) {
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
                    <h3 className="font-semibold text-zinc-300">Escalation events</h3>
                    <p className="mt-2 text-[11px] text-zinc-500">
                        Not implemented yet — the direct-video escalation tool (System Design §5, roadmap X1b)
                        has not shipped, so there is nothing to inspect here.
                    </p>
                </section>
            </div>
        </div>
    );
}
