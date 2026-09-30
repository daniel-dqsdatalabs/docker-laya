"""Validated request and response contracts for the HTTP API."""

import secrets
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelName = Literal["english", "multilingual", "typed-decisions"]
State = str | dict[str, Any] | list[Any]
InstructionValue = str | dict[str, Any] | list[Any]
CriteriaValue = Any  # Criteria accept any JSON value, including null.
PresetName = Literal["triage", "email", "guard", "moderation", "router"]


class ChoiceQuestion(BaseModel):
    """Pick one labelled option, e.g. a routing department."""

    type: Literal["choice"]
    instructions: InstructionValue = Field(..., description="What to decide, phrased as a question.")
    criteria: dict[str, CriteriaValue] | list[CriteriaValue] | None = Field(
        default=None,
        description=(
            "Option label -> description, or a plain list of labels. Descriptions may be "
            "strings or any JSON value (rendered as compact JSON)."
        ),
        examples=[{"billing": "invoices, payments, refunds", "technical": ["bugs", "outages"]}],
    )


class ScoreQuestion(BaseModel):
    """Rate on an ordered scale, e.g. urgency."""

    type: Literal["score"]
    instructions: InstructionValue
    criteria: list[CriteriaValue] = Field(
        ...,
        description="Ordered levels, lowest first. Each may be any JSON value.",
        examples=[["low", {"level": "high", "sla_minutes": 60}]],
    )


class NoulQuestion(BaseModel):
    """Yes/no decision without a learned neutral class (n-o-u-l)."""

    type: Literal["noul"]
    instructions: InstructionValue
    criteria: dict[str, CriteriaValue] | None = Field(
        default=None,
        description="Optional `false`/`true` descriptions (any JSON value).",
        examples=[{"true": "phishing or fraud", "false": "a legitimate email"}],
    )


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


QUESTION_EXAMPLE: dict[str, Any] = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs and outages",
            "sales": "new purchases",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical"],
    },
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
}


class PredictRequest(BaseModel):
    """One state with either explicit typed questions or a built-in preset."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "state": "I was billed twice. Please refund the duplicate today.",
                    "questions": QUESTION_EXAMPLE,
                },
                {"state": "Hi, we were billed twice for March.", "preset": "triage"},
            ]
        }
    )

    state: State = Field(..., description="Text, JSON object, or conversation turns to analyse.")
    questions: dict[str, Question] | None = Field(
        default=None, description="Map of question id -> typed question definition."
    )
    preset: PresetName | None = Field(default=None, description="Built-in question set to use instead of `questions`.")
    model: ModelName | None = Field(default=None, description="Pin a checkpoint; omit to auto-route by language.")

    @model_validator(mode="after")
    def _exactly_one_source(self) -> PredictRequest:
        """Reject requests that provide neither question source or both sources at once."""
        if (self.questions is None) == (self.preset is None):
            raise ValueError("Provide exactly one of `questions` or `preset`")
        return self


class BulkItem(BaseModel):
    """One state with optional per-item questions and model override."""

    state: State
    questions: dict[str, Question] | None = Field(
        default=None, description="Overrides the request-level questions for this item."
    )
    model: ModelName | None = Field(default=None, description="Overrides the request-level model for this item.")


class BulkPredictRequest(BaseModel):
    """A batch with shared questions or optional per-item question and model overrides."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "states": [
                        "I was billed twice. Please refund the duplicate today.",
                        "The app crashes on launch.",
                    ],
                    "questions": QUESTION_EXAMPLE,
                },
                {
                    "items": [
                        {"state": "I was billed twice."},
                        {"state": "The app crashes.", "model": "english"},
                    ],
                    "preset": "triage",
                },
            ]
        }
    )

    states: list[State] | None = Field(
        default=None,
        min_length=1,
        description="States to classify with the shared `questions`/`preset`.",
    )
    questions: dict[str, Question] | None = Field(default=None, description="Questions applied to each state.")
    preset: PresetName | None = Field(default=None, description="Preset questions for each state.")
    model: ModelName | None = Field(default=None, description="Pin a checkpoint for all states; omit to auto-route.")
    items: list[BulkItem] | None = Field(
        default=None,
        min_length=1,
        description="Per-state items, each optionally overriding questions and model.",
    )

    @model_validator(mode="after")
    def _shape(self) -> BulkPredictRequest:
        """Validate exclusive batch shapes and the shared question source required by states."""
        if self.items and self.states:
            raise ValueError("Provide either `items` or `states`, not both")
        if not self.items and not self.states:
            raise ValueError("Provide `items` or `states`")
        if self.questions is not None and self.preset is not None:
            raise ValueError("Provide either `questions` or `preset`, not both")
        if self.states and self.questions is None and self.preset is None:
            raise ValueError("`states` requires `questions` or `preset`")
        return self


class ActionResult(BaseModel):
    """Probability that the model took an action for a question."""

    act_probability: float = Field(..., description="Probability the model acted at all.")


class SystemOneChoiceAnswer(BaseModel):
    """Choice result without native Laya action metadata."""

    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


class SystemOneScoreAnswer(BaseModel):
    """Ordered score without native Laya action metadata."""

    type: Literal["score"]
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


class SystemOneNoulAnswer(BaseModel):
    """Binary decision in the reduced SystemOne response format."""

    type: Literal["noul"]
    noul: float


class ChoiceAnswer(SystemOneChoiceAnswer):
    """Selected option and calibrated probabilities for a choice question."""

    action: ActionResult


class ScoreAnswer(SystemOneScoreAnswer):
    """Expected score, scale legend and probabilities for an ordered question."""

    score: float = Field(..., description="Expected value on the 0-based criteria scale.")
    action: ActionResult


class NoulAnswer(SystemOneNoulAnswer):
    """Binary decision value, confidence and action probability."""

    confidence: float
    action: ActionResult


Answer = Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    """Input and output token counts reported by the model."""

    input_tokens: int
    output_tokens: int


class Routing(BaseModel):
    """Chosen checkpoint and optional routing details supplied by Laya."""

    model_config = ConfigDict(extra="allow")

    model: str
    reason: str | None = None


class PredictResponse(BaseModel):
    """Validated native prediction, preserving additional fields returned by Laya."""

    model_config = ConfigDict(extra="allow")

    model: str
    answers: dict[str, Answer]
    usage: Usage
    routing: Routing | None = Field(default=None, description="How the checkpoint was chosen (auto-route or pinned).")


SystemOneAnswer = Annotated[
    SystemOneNoulAnswer | SystemOneChoiceAnswer | SystemOneScoreAnswer,
    Field(discriminator="type"),
]


class SystemOneRequest(BaseModel):
    """SystemOne-compatible input accepting model aliases and optional client metadata."""

    model_config = ConfigDict(extra="ignore")

    state: State = Field(..., description="The content to evaluate: a string, dict, or list.")
    model: str = Field(
        ...,
        description="System One model ID, e.g. laya-english, laya-multilingual, typed-decisions.",
    )
    questions: dict[str, Question] = Field(..., description="Typed questions map.")
    session_id: str | None = Field(default=None, description="Optional session identifier.")
    user: str | None = Field(default=None, description="Optional user identifier.")


class SystemOneResponse(BaseModel):
    """SystemOne-compatible prediction with a generated request identifier."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=lambda: f"gen-laya-{secrets.token_hex(12)}")
    model: str
    provider: str = "Laya"
    answers: dict[str, SystemOneAnswer]
    usage: Usage


class BulkItemResult(BaseModel):
    """Either a validated prediction or an isolated error for one batch item."""

    ok: bool
    result: PredictResponse | None = None
    error: str | None = None


class BulkPredictResponse(BaseModel):
    """Ordered results and item count for a completed batch."""

    count: int
    results: list[BulkItemResult]


class HealthResponse(BaseModel):
    """Model readiness, device and enabled authentication methods."""

    status: str
    model_loaded: bool
    device: str
    auth_enabled: bool
    auth_methods: list[str]
    models: list[str] = Field(..., description="Checkpoints currently resident.")
    available_models: list[str]


class ModelsResponse(BaseModel):
    """Available checkpoints and the subset currently resident in memory."""

    available: list[str]
    loaded: list[str]


class DetectRequest(BaseModel):
    """Text or structured state submitted for language detection."""

    state: State


class EmailStateRequest(BaseModel):
    """Email content and options used to construct a prediction state."""

    subject: str = ""
    body: str
    sender: str | None = None
    clean: bool = Field(default=True, description="Strip quoted history, signatures, disclaimers.")
