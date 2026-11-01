"""Data models shared across the pipeline.

The same pydantic models double as structured-output schemas for the LLM,
so every LLM call returns validated objects rather than free text.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EntityKind = Literal["person", "organization", "group", "place", "concept", "event"]
ActionKind = Literal["post", "reply", "repost", "like", "idle"]

# Entity kinds that can act in the simulation (become agents).
ACTOR_KINDS: set[str] = {"person", "organization", "group"}


# ---------------------------------------------------------------- graph


class Entity(BaseModel):
    name: str
    kind: EntityKind
    summary: str = ""


class Relation(BaseModel):
    source: str
    target: str
    label: str
    evidence: str = ""


class Extraction(BaseModel):
    """LLM output for one chunk of seed text."""

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


# ---------------------------------------------------------------- personas


class Persona(BaseModel):
    id: str
    name: str
    kind: EntityKind
    bio: str
    traits: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    # -1.0 strongly against the question's outcome, +1.0 strongly for it.
    stance: float = Field(0.0, ge=-1.0, le=1.0)
    # How much attention this agent's posts get, 0..1.
    influence: float = Field(0.5, ge=0.0, le=1.0)
    # Probability of acting in a given round, 0..1.
    activity: float = Field(0.6, ge=0.0, le=1.0)
    voice: str = "plain"


class PersonaDraft(BaseModel):
    """LLM output when turning a graph entity into a persona (no id yet)."""

    bio: str
    traits: list[str]
    goals: list[str]
    stance: float = Field(ge=-1.0, le=1.0)
    influence: float = Field(ge=0.0, le=1.0)
    activity: float = Field(ge=0.0, le=1.0)
    voice: str


# ---------------------------------------------------------------- simulation


class AgentDecision(BaseModel):
    """LLM output: what one agent does in one round."""

    action: ActionKind
    target_post_id: str | None = None
    content: str = ""
    # Updated stance after reading the feed, -1..1.
    stance: float = Field(ge=-1.0, le=1.0)
    reflection: str = ""


class Post(BaseModel):
    id: str
    round: int
    author_id: str
    author_name: str
    kind: Literal["post", "reply", "repost", "event"]
    content: str
    parent_id: str | None = None
    likes: list[str] = Field(default_factory=list)
    reposts: list[str] = Field(default_factory=list)
    stance: float = 0.0

    @property
    def engagement(self) -> int:
        return len(self.likes) + 2 * len(self.reposts)


class Injection(BaseModel):
    """A variable dropped into the world from the god's-eye view."""

    round: int
    text: str


class RoundStats(BaseModel):
    round: int
    posts: int
    replies: int
    reposts: int
    likes: int
    idle: int
    mean_stance: float
    support_share: float
    oppose_share: float
    neutral_share: float


# ---------------------------------------------------------------- report


class Scenario(BaseModel):
    name: str
    probability: float = Field(ge=0.0, le=1.0)
    description: str


class Report(BaseModel):
    headline: str
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    key_drivers: list[str]
    turning_points: list[str]
    scenarios: list[Scenario]
    risks: list[str]
    markdown: str = ""
