"""Pydantic models: input items and the JSON that each LLM role must return."""
from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

FORMATS = ("mcq_4", "numeric", "staged", "short_text", "rubric_task", "code")
KEYED_FORMATS = ("mcq_4", "numeric", "staged")


class NumericKey(BaseModel):
    value: float
    unit: str = ""
    tolerance: Optional[float] = None


class Option(BaseModel):
    id: str
    text: str
    misconception_id: str = ""
    failed_layer: str = ""
    rationale: str = ""
    correct: Optional[bool] = None


class Stage(BaseModel):
    stage_id: str
    prompt: str
    answer_key: Union[NumericKey, str, None] = None


class Criterion(BaseModel):
    criterion_id: str
    text: str
    weight: float = 1.0
    layer: str = ""
    anchor_pass: str = ""
    anchor_fail: str = ""


class Item(BaseModel):
    item_id: str
    format: str
    stem: str
    options: list[Option] = Field(default_factory=list)
    answer_key: Union[NumericKey, str, None] = None
    stages: list[Stage] = Field(default_factory=list)
    rubric: list[Criterion] = Field(default_factory=list)
    model_answer: str = ""
    topic: str = ""
    course_code: str = "_ALL"
    course_title: str = ""
    syllabus_topics: list[str] = Field(default_factory=list)
    skill: str = ""
    bloom_level: Optional[int] = None
    difficulty_band: str = ""
    source: str = ""
    explanation: str = ""
    hints: Any = None
    textbook_ref: str = ""
    human_key_correct: Optional[bool] = None
    row_number: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)

    def eligible(self, metric: str) -> bool:
        if metric == "m1":
            return self.format in KEYED_FORMATS
        if metric == "m2":
            return bool(self.rubric)
        if metric == "m3":
            return bool(self.topic.strip())
        raise ValueError(metric)


# ---------------------------------------------------------------- LLM outputs (lenient: extra keys ignored)

class _Out(BaseModel):
    model_config = ConfigDict(extra="ignore")


class StageAnswer(_Out):
    value: Any = None
    unit: str = ""

    @field_validator("unit", mode="before")
    @classmethod
    def _unit(cls, v: Any) -> str:
        return "" if v is None else str(v)


class OptionVerdict(_Out):
    id: str
    correct: bool = False
    reason: str = ""


class SolverOut(_Out):
    answer: Any = None
    unit: str = ""
    stage_answers: dict[str, StageAnswer] = Field(default_factory=dict)
    option_verdicts: list[OptionVerdict] = Field(default_factory=list)
    ambiguous: bool = False
    ambiguity_reason: str = ""
    working: str = ""

    @field_validator("unit", "ambiguity_reason", "working", mode="before")
    @classmethod
    def _s(cls, v: Any) -> str:
        return "" if v is None else str(v)

    @field_validator("stage_answers", mode="before")
    @classmethod
    def _stages(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return {}
        return {str(k): (x if isinstance(x, dict) else {"value": x}) for k, x in v.items()}


class VerifierOut(_Out):
    key_correct: bool
    correct_answer: Any = ""
    reason: str = ""


class ReferenceOut(_Out):
    answer: str


class MutationOut(_Out):
    criterion_id: str = ""
    mutated_answer: str
    what_was_broken: str = ""
    edit_size: str = ""


class ParaphraseOut(_Out):
    paraphrased_answer: str


class MutationValidOut(_Out):
    target_violated: bool
    others_preserved: bool
    fluent: bool
    note: str = ""


class ParaphraseValidOut(_Out):
    meaning_preserved: bool
    all_criteria_still_met: bool
    note: str = ""


class JudgeVerdict(_Out):
    criterion_id: str
    verdict: str
    quote: str = ""
    confidence: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def _v(cls, v: Any) -> str:
        return str(v).strip().lower()

    @field_validator("quote", "confidence", mode="before")
    @classmethod
    def _s(cls, v: Any) -> str:
        return "" if v is None else str(v)


class JudgeOut(_Out):
    verdicts: list[JudgeVerdict] = Field(default_factory=list)
    injection_detected: bool = False


class RankedTopic(_Out):
    topic: str
    confidence: float = 0.0

    @field_validator("confidence", mode="before")
    @classmethod
    def _c(cls, v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0


class TopicOut(_Out):
    ranked: list[RankedTopic] = Field(default_factory=list)
    reason: str = ""
