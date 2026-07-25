import { describe, expect, it } from "vitest";

import {
  applyProposalAccept,
  applyProposalAcceptAll,
  applyProposalReject,
  applyProposalRejectAll,
  applyProposalUndo,
  mergeNewProposals,
} from "./inspector";

function trimProposal(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    id: "p1",
    action: "trim_video" as const,
    operation: "remove_segment" as const,
    start_sec: 2,
    end_sec: 4,
    reason: "boring",
    confidence: 0.8,
    status: "pending" as const,
    ...overrides,
  };
}

function speedProposal(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    id: "p2",
    action: "speed_video" as const,
    operation: "apply_speed_range" as const,
    start_sec: 5,
    end_sec: 7,
    reason: "speed up",
    confidence: 0.6,
    speed_multiplier: 2,
    status: "pending" as const,
    ...overrides,
  };
}

describe("applyProposalAccept", () => {
  it("applies a pending trim proposal to trimRanges and marks it accepted", () => {
    const result = applyProposalAccept([trimProposal()], [], [], "p1");
    expect(result.trimRanges).toEqual([{ start: 2, end: 4 }]);
    expect(result.proposals[0].status).toBe("accepted");
  });

  it("applies a pending speed proposal to speedRanges", () => {
    const result = applyProposalAccept([speedProposal()], [], [], "p2");
    expect(result.speedRanges).toEqual([{ start: 5, end: 7, speed: 2 }]);
    expect(result.proposals[0].status).toBe("accepted");
  });

  it("is a no-op for a proposal that is not pending", () => {
    const result = applyProposalAccept([trimProposal({ status: "rejected" })], [], [], "p1");
    expect(result.trimRanges).toEqual([]);
    expect(result.proposals[0].status).toBe("rejected");
  });

  it("is a no-op for an unknown id", () => {
    const result = applyProposalAccept([trimProposal()], [], [], "missing");
    expect(result.trimRanges).toEqual([]);
  });
});

describe("applyProposalReject / applyProposalRejectAll", () => {
  it("marks a single proposal rejected without touching ranges", () => {
    const result = applyProposalReject([trimProposal()], "p1");
    expect(result[0].status).toBe("rejected");
  });

  it("rejects only pending proposals, leaving accepted ones alone", () => {
    const result = applyProposalRejectAll([trimProposal(), speedProposal({ status: "accepted" })]);
    expect(result[0].status).toBe("rejected");
    expect(result[1].status).toBe("accepted");
  });
});

describe("applyProposalUndo", () => {
  it("removes the accepted proposal's range and flips it back to pending", () => {
    const accepted = trimProposal({ status: "accepted" });
    const result = applyProposalUndo([accepted], [{ start: 2, end: 4 }], [], "p1");
    expect(result.trimRanges).toEqual([]);
    expect(result.proposals[0].status).toBe("pending");
  });

  it("only removes the matching range, not unrelated ones", () => {
    const accepted = trimProposal({ status: "accepted" });
    const result = applyProposalUndo(
      [accepted],
      [
        { start: 2, end: 4 },
        { start: 10, end: 12 },
      ],
      [],
      "p1"
    );
    expect(result.trimRanges).toEqual([{ start: 10, end: 12 }]);
  });

  it("is a no-op for a proposal that is not accepted", () => {
    const result = applyProposalUndo([trimProposal()], [{ start: 2, end: 4 }], [], "p1");
    expect(result.trimRanges).toEqual([{ start: 2, end: 4 }]);
  });
});

describe("applyProposalAcceptAll", () => {
  it("applies every pending proposal in one commit", () => {
    const result = applyProposalAcceptAll([trimProposal(), speedProposal()], [], []);
    expect(result.trimRanges).toEqual([{ start: 2, end: 4 }]);
    expect(result.speedRanges).toEqual([{ start: 5, end: 7, speed: 2 }]);
    expect(result.proposals.every((p) => p.status === "accepted")).toBe(true);
  });

  it("leaves already-accepted or rejected proposals untouched", () => {
    const result = applyProposalAcceptAll(
      [trimProposal({ id: "p1", status: "accepted" }), speedProposal({ id: "p2", status: "rejected" })],
      [],
      []
    );
    expect(result.trimRanges).toEqual([]);
    expect(result.speedRanges).toEqual([]);
    expect(result.proposals[0].status).toBe("accepted");
    expect(result.proposals[1].status).toBe("rejected");
  });
});

describe("mergeNewProposals", () => {
  it("drops old pending proposals when a new plan call arrives", () => {
    const stale = trimProposal({ id: "stale", status: "pending" });
    const incoming = [trimProposal({ id: "fresh" })];
    const result = mergeNewProposals([stale], incoming);
    expect(result.map((p) => p.id)).toEqual(["fresh"]);
  });

  it("keeps already-accepted and already-rejected proposals from before", () => {
    const accepted = trimProposal({ id: "kept-accepted", status: "accepted" });
    const rejected = speedProposal({ id: "kept-rejected", status: "rejected" });
    const incoming = [trimProposal({ id: "fresh" })];
    const result = mergeNewProposals([accepted, rejected], incoming);
    expect(result.map((p) => p.id)).toEqual(["kept-accepted", "kept-rejected", "fresh"]);
  });
});
