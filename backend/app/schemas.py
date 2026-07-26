from typing import Literal, Optional

from pydantic import BaseModel, Field


class SpriteFrame(BaseModel):
    index: int
    timestamp_sec: float
    row: int
    col: int


class SpriteSheet(BaseModel):
    sheet_index: int
    image_url: str
    image_width: int
    image_height: int
    tile_width: int
    tile_height: int
    start_time_sec: float
    end_time_sec: float
    frames: list[SpriteFrame]


class SpriteAnalysisResponse(BaseModel):
    duration_sec: float
    interval_sec: float
    columns: int
    rows: int
    total_frames: int
    sprite_job_id: str = ""
    sheets: list[SpriteSheet]


class TokenEstimateRequest(BaseModel):
    duration_sec: float = Field(gt=0)
    interval_sec: float = Field(default=0.25, gt=0)
    columns: int = Field(default=8, gt=0)
    rows: int = Field(default=8, gt=0)
    thumb_width: int = Field(default=256, ge=64)


class TokenEstimateResponse(BaseModel):
    duration_sec: float
    direct_video_tokens_est: int
    sprite_tokens_est: int
    total_frames: int
    sheet_count: int
    recommendation: str
    notes: list[str]


class EditSuggestion(BaseModel):
    id: str
    action: Literal["trim_video", "speed_video"] = "trim_video"
    operation: Literal["remove_segment", "extract_range", "apply_speed_range"] = "remove_segment"
    start_sec: float
    end_sec: float
    reason: str
    confidence: float = 0.5
    speed_multiplier: Optional[float] = None


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class SpeedRangeInput(BaseModel):
    start: float
    end: float
    speed: float


class TrimRange(BaseModel):
    start: float
    end: float


class ConversationSummaryRequest(BaseModel):
    # Only the turns that just fell out of the raw chat_history window (P3-5) —
    # a rolling summary, not the whole session re-summarized every call.
    older_turns: list[ChatTurn] = Field(default_factory=list)
    previous_summary: Optional[str] = None


class ConversationSummaryResponse(BaseModel):
    summary: str
    model: str


class AgentPlanRequest(BaseModel):
    prompt: str = Field(min_length=1)
    duration_sec: float = Field(gt=0)
    sprite_interval_sec: float = Field(gt=0)
    total_frames: int = Field(gt=0)
    sheets_count: int = Field(gt=0)
    sprite_job_id: Optional[str] = None
    chat_history: list[ChatTurn] = Field(default_factory=list)
    conversation_summary: Optional[str] = None
    trim_ranges: list[TrimRange] = Field(default_factory=list)
    speed_ranges: list[SpeedRangeInput] = Field(default_factory=list)
    # Dev-panel-only override (Design Handoff Part 3): live-adjustable escalation
    # threshold for this call, not persisted; falls back to the server default.
    escalation_confidence_threshold: Optional[float] = Field(default=None, ge=0, le=1)


class EscalationEvent(BaseModel):
    window_start_sec: float
    window_end_sec: float
    trigger: Literal["user_cue", "low_confidence"]
    confidence_before: float
    tokens_used: Optional[int] = None


class AgentPlanResponse(BaseModel):
    plan_id: str
    reasoning: str
    proposals: list[EditSuggestion]
    model: str
    strategy: str
    tokens_used: Optional[int] = None
    escalation: Optional[EscalationEvent] = None


class ExportResponse(BaseModel):
    output_url: str
    output_name: str
    removed_ranges_count: int
