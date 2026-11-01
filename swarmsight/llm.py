"""LLM backends.

Two interchangeable backends implement the same small interface:

* ``ClaudeLLM`` calls Claude through the official Anthropic SDK with
  structured outputs, so each call returns a validated pydantic object.
* ``MockLLM`` is deterministic and offline. It powers the test suite and the
  ``--mock`` demo mode, so the whole pipeline runs without an API key.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
from typing import Protocol, TypeVar

from pydantic import BaseModel

from .schemas import (
    AgentDecision,
    Entity,
    Extraction,
    PersonaDraft,
    Relation,
    Report,
    Scenario,
)

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-opus-5-5"


class LLM(Protocol):
    name: str

    def structured(self, system: str, prompt: str, schema: type[T], *, task: str) -> T: ...

    def text(self, system: str, messages: list[dict], *, task: str) -> str: ...


class LLMError(RuntimeError):
    pass


# ====================================================================== Claude


class ClaudeLLM:
    """Claude via the Anthropic SDK.

    ``task`` picks the effort level: bulk per-agent calls run at ``low`` so a
    many-agent simulation stays affordable; the report runs at ``high``.
    """

    EFFORT = {"extract": "medium", "persona": "low", "act": "low", "report": "high", "chat": "low"}

    def __init__(self, model: str | None = None):
        import anthropic  # imported lazily so mock mode needs no SDK

        self.model = model or os.environ.get("SWARMSIGHT_MODEL", DEFAULT_MODEL)
        self.name = f"claude:{self.model}"
        self._client = anthropic.Anthropic()

    def structured(self, system: str, prompt: str, schema: type[T], *, task: str) -> T:
        import anthropic

        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            # Re-route automatically if a safety classifier declines a request.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={
                "effort": self.EFFORT.get(task, "medium"),
                "format": {"type": "json_schema",
                           "schema": anthropic.transform_schema(schema.model_json_schema())},
            },
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise LLMError(f"Claude declined the {task} request")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"{task} output hit max_tokens")
        raw = "".join(b.text for b in response.content if b.type == "text")
        try:
            return schema.model_validate_json(raw)
        except ValueError as exc:
            raise LLMError(f"{task} returned invalid JSON: {exc}") from exc

    def text(self, system: str, messages: list[dict], *, task: str) -> str:
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": self.EFFORT.get(task, "medium")},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        )
        if response.stop_reason == "refusal":
            raise LLMError(f"Claude declined the {task} request")
        return "".join(b.text for b in response.content if b.type == "text").strip()


# ======================================================================== Mock

_STOP = {
    "The", "A", "An", "In", "On", "At", "By", "For", "And", "But", "Or", "If", "When", "While",
    "This", "That", "These", "Those", "It", "Its", "As", "After", "Before", "Since", "With",
    "Without", "From", "To", "Of", "Into", "Over", "Under", "Many", "Some", "Most", "All",
    "Several", "Meanwhile", "However", "Analysts", "Critics", "Supporters", "Officials",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Today",
    "Yesterday", "Tomorrow", "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December", "We", "They", "He", "She", "I",
    "Its", "His", "Her", "Their", "Local",
}
_TITLES = {"Mayor", "Councillor", "Councilor", "Professor", "Dr", "Mr", "Ms", "Mrs", "Senator",
           "Minister", "President", "Governor", "Judge", "Chief", "Director", "Union",
           "Consultant", "Engineer", "Analyst", "Chairman", "Chairperson", "Founder", "Officer", "Shri", "Smt"}
_ORG_HINTS = ("Inc", "Corp", "Council", "Ministry", "Bank", "Union", "Agency", "Board",
              "Committee", "Party", "University", "Association", "Group", "Times", "News",
              "Institute", "Authority", "Company", "Labs", "Fund", "Department", "Federation")
_GROUP_HINTS = ("Residents", "Workers", "Students", "Parents", "Voters", "Farmers", "Drivers",
                "Investors", "Customers", "Members", "Owners", "Coalition", "Alliance", "Network")
_PLACE_HINTS = ("City", "County", "Street", "River", "Valley", "District", "Province", "Port",
                "Harbor", "Bay", "Park", "Square")
# Names that look like proper nouns but are metrics, units, rules or documents, never actors.
_CONCEPT_HINTS = {"Rule", "Rules", "Act", "Policy", "Standard", "Standards", "Rate", "Return", "Returns",
                  "Cost", "Costs", "Revenue", "Margin", "Margins", "Capex", "Opex", "Price", "Prices",
                  "Tariff", "Permit", "License", "Licence", "Plan", "Report", "Study", "Test", "Project",
                  "Projected", "Process", "Scheme", "Tax", "Duty", "Loan", "Grant", "Budget", "Forecast",
                  "Profit", "Loss", "Sales", "Volume", "Capacity", "Demand", "Supply", "Stage", "Phase"}
_METRIC_ACRONYMS = {"IRR", "NPV", "ROI", "ROE", "ROCE", "EBITDA", "EBIT", "CAGR", "DSCR", "GST", "VAT",
                    "KPI", "OKR", "FY", "YOY", "QOQ", "MT", "TPD", "TPA", "KG", "KL", "MW", "KW", "KWH",
                    "INR", "USD", "EUR", "GBP", "ISI", "ISO", "LOI", "MOU", "NDA", "SOP", "PPE", "COD",
                    "BOD", "TDS", "PH", "PDF", "CEO", "CFO", "CTO", "HQ", "AM", "PM", "TBD", "NA", "OK"}
# Inflected forms are listed explicitly; the mock does no stemming.
_POS = {"support", "supports", "supported", "supporting", "back", "backs", "backed", "welcome",
        "welcomes", "welcomed", "benefit", "benefits", "growth", "grow", "grows", "praise", "praised",
        "approve", "approves", "approved", "approval", "boost", "boosts", "jobs", "endorse", "endorses",
        "endorsed", "favor", "favour", "favourable", "favorable", "agree", "agrees", "agreed", "pass",
        "passes", "passed", "meet", "meets", "met", "compliant", "grant", "grants", "granted", "clear",
        "clears", "cleared", "success", "successful", "succeeds", "profit", "profitable", "viable",
        "strong", "secure", "secured", "signed", "win", "wins", "won", "gain", "gains", "positive",
        "opportunity", "interested", "commit", "commits", "committed", "confirm", "confirms", "confirmed"}
_NEG = {"oppose", "opposes", "opposed", "against", "protest", "protests", "criticize", "criticized",
        "criticise", "criticised", "concern", "concerns", "concerned", "risk", "risks", "risky", "reject",
        "rejects", "rejected", "rejection", "fear", "fears", "feared", "warn", "warns", "warned", "harm",
        "harmful", "cost", "costs", "costly", "anger", "angry", "lawsuit", "sue", "sues", "block", "blocks",
        "blocked", "delay", "delays", "delayed", "fail", "fails", "failed", "failure", "deny", "denies",
        "denied", "refuse", "refuses", "refused", "penalty", "penalties", "violation", "violates", "loss",
        "losses", "decline", "declines", "declined", "weak", "uncertain", "uncertainty", "shortage",
        "negative", "problem", "problems", "dispute", "doubt", "doubts", "unviable", "withdraw", "withdrew"}


def _seed_int(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:12], 16)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _lexicon_score(text: str) -> float:
    words = re.findall(r"[a-z']+", text.lower())
    pos = sum(w in _POS for w in words)
    neg = sum(w in _NEG for w in words)
    return 0.0 if pos == neg else (pos - neg) / (pos + neg)


def _classify(name: str) -> str:
    words = name.split()
    last = words[-1]
    if name in _METRIC_ACRONYMS or any(w in _CONCEPT_HINTS or w in _METRIC_ACRONYMS for w in words):
        return "concept"
    if any(h in name for h in _ORG_HINTS) or (name.isupper() and len(name) >= 3):
        return "organization"
    if last in _GROUP_HINTS or name.endswith("s") and last in _GROUP_HINTS:
        return "group"
    if any(h in name for h in _PLACE_HINTS):
        return "place"
    if len(name.split()) in (2, 3) and all(p[:1].isupper() for p in name.split()):
        return "person"
    return "concept"


class MockLLM:
    """Deterministic stand-in for an LLM.

    It is not a language model; it uses simple heuristics so the pipeline can
    be exercised end to end, offline, with reproducible results.
    """

    name = "mock"

    def structured(self, system: str, prompt: str, schema: type[T], *, task: str) -> T:
        handler = {
            Extraction: self._extract,
            PersonaDraft: self._persona,
            AgentDecision: self._decide,
            Report: self._report,
        }.get(schema)
        if handler is None:
            raise LLMError(f"MockLLM has no handler for {schema.__name__}")
        return handler(prompt)  # type: ignore[return-value]

    def text(self, system: str, messages: list[dict], *, task: str) -> str:
        question = messages[-1]["content"] if messages else ""
        name = _field(system, "NAME") or "I"
        stance = float(_field(system, "STANCE") or 0)
        mood = "in favour of" if stance > 0.2 else "against" if stance < -0.2 else "undecided about"
        memory = _field(system, "LAST_MEMORY")
        tail = f" The last thing on my mind: {memory}" if memory else ""
        return f"Speaking as {name}: I'm {mood} this. You asked \"{question[:120]}\".{tail}"

    # -- extraction -------------------------------------------------------

    def _extract(self, prompt: str) -> Extraction:
        text = prompt.split("TEXT:", 1)[-1]
        text = re.sub(r"^#+ .*$", "", text, flags=re.M).replace("*", "")  # drop markdown titles/emphasis
        pattern = re.compile(r"\b([A-Z][a-zA-Z&'-]+(?:\s+(?:of|the|for|and|[A-Z][a-zA-Z&'-]+))*)")
        entities: dict[str, Entity] = {}
        per_sentence: list[list[str]] = []
        for sentence in _sentences(text):
            found = []
            for match in pattern.finditer(sentence):
                name = re.sub(r"\s+(of|the|for|and)$", "", match.group(1)).strip()
                words = name.split()
                while words and (words[0] in _STOP or (words[0] in _TITLES and len(words) > 1)):
                    words = words[1:]
                name = " ".join(words)
                if len(name) < 3 or name in _STOP:
                    continue
                kind = _classify(name)
                if name not in entities:
                    entities[name] = Entity(name=name, kind=kind, summary=sentence[:200])
                found.append(name)
            per_sentence.append(found)
        # Drop one-word fragments of longer names ("Union" from "Drivers Union").
        long_words = {w for n in entities if len(n.split()) > 1 for w in n.split()}
        for name in [n for n in entities if len(n.split()) == 1 and n in long_words]:
            del entities[name]
        per_sentence = [[n for n in names if n in entities] for names in per_sentence]
        relations = []
        for sentence, names in zip(_sentences(text), per_sentence):
            for a, b in zip(names, names[1:]):
                if a != b:
                    verb = next((w for w in re.findall(r"[a-z]+", sentence.lower()) if w in _POS | _NEG), "mentioned with")
                    relations.append(Relation(source=a, target=b, label=verb, evidence=sentence[:200]))
        return Extraction(entities=list(entities.values()), relations=relations)

    # -- personas ---------------------------------------------------------

    def _persona(self, prompt: str) -> PersonaDraft:
        name = _field(prompt, "NAME") or "Agent"
        kind = _field(prompt, "KIND") or "person"
        context = prompt.split("CONTEXT:", 1)[-1]
        rng = random.Random(_seed_int("persona", name))
        lean = _lexicon_score(context)
        stance = max(-1.0, min(1.0, 0.6 * lean + rng.uniform(-0.5, 0.5)))
        trait_pool = ["pragmatic", "skeptical", "optimistic", "data-driven", "outspoken", "cautious",
                      "community-minded", "ambitious", "contrarian", "loyal", "analytical", "emotional"]
        traits = rng.sample(trait_pool, 3)
        influence = {"organization": 0.75, "group": 0.6}.get(kind, 0.4) + rng.uniform(-0.15, 0.2)
        return PersonaDraft(
            bio=f"{name} ({kind}). {context.strip()[:220]}",
            traits=traits,
            goals=[f"protect the interests of {name}", "be heard on the issue"],
            stance=round(stance, 2),
            influence=round(max(0.05, min(1.0, influence)), 2),
            activity=round(rng.uniform(0.45, 0.95), 2),
            voice=rng.choice(["formal", "plain", "punchy", "measured", "passionate"]),
        )

    # -- per-round decisions ----------------------------------------------

    def _decide(self, prompt: str) -> AgentDecision:
        agent_id = _field(prompt, "AGENT_ID") or "a"
        name = _field(prompt, "NAME") or "Agent"
        round_no = _field(prompt, "ROUND") or "0"
        stance = float(_field(prompt, "STANCE") or 0)
        activity = float(_field(prompt, "ACTIVITY") or 0.6)
        rng = random.Random(_seed_int("decide", agent_id, round_no))

        feed = re.findall(r"^\[(p\d+)\] .*?\(stance ([-0-9.]+), engagement (\d+)\): (.*)$",
                          prompt, flags=re.M)
        injected = _field(prompt, "WORLD_EVENT")

        # Social influence: drift toward the engagement-weighted feed mood,
        # plus a nudge from any injected event.
        pull = 0.0
        if feed:
            weights = [1 + int(e) for _, _, e, _ in feed]
            pull = sum(float(s) * w for (_, s, _, _), w in zip(feed, weights)) / sum(weights)
        shock = _lexicon_score(injected) * 0.25 if injected else 0.0
        new_stance = stance + 0.12 * (pull - stance) + shock + rng.uniform(-0.05, 0.05)
        new_stance = round(max(-1.0, min(1.0, new_stance)), 3)

        if rng.random() > activity:
            return AgentDecision(action="idle", stance=new_stance, reflection="Watched quietly.")

        side = "support" if new_stance > 0.2 else "oppose" if new_stance < -0.2 else "am unsure about"
        if feed and rng.random() < 0.55:
            agree = [p for p in feed if (float(p[1]) > 0) == (new_stance > 0)]
            pid, _, _, text = rng.choice(agree or feed)
            roll = rng.random()
            if roll < 0.4:
                return AgentDecision(action="like", target_post_id=pid, stance=new_stance,
                                     reflection=f"Agreed with {pid}.")
            if roll < 0.6:
                return AgentDecision(action="repost", target_post_id=pid, stance=new_stance,
                                     reflection=f"Amplified {pid}.")
            verb = "Exactly right" if (float(dict((p[0], p[1]) for p in feed)[pid]) > 0) == (new_stance > 0) else "I disagree"
            return AgentDecision(action="reply", target_post_id=pid, stance=new_stance,
                                 content=f"{verb}. As {name}, I {side} this.",
                                 reflection=f"Replied to {pid}.")
        hook = f" after the news that {injected.rstrip('.')}" if injected else ""
        return AgentDecision(action="post", stance=new_stance,
                             content=f"{name} here: we {side} it{hook}.",
                             reflection="Shared a view.")

    # -- report -----------------------------------------------------------

    def _report(self, prompt: str) -> Report:
        question = _field(prompt, "QUESTION") or "the question"
        start = float(_field(prompt, "START_STANCE") or 0)
        end = float(_field(prompt, "END_STANCE") or 0)
        support = float(_field(prompt, "END_SUPPORT") or 0)
        oppose = float(_field(prompt, "END_OPPOSE") or 0)
        leaning = "likely" if end > 0.15 else "unlikely" if end < -0.15 else "too close to call"
        confidence = round(min(0.9, 0.35 + abs(end) * 0.6), 2)
        p_yes = round(max(0.05, min(0.95, 0.5 + end / 2)), 2)
        return Report(
            headline=f"[Mock demo, not AI] Outcome looks {leaning}",
            prediction=(f"For \"{question}\": the simulated population ended at mean stance {end:+.2f} "
                        f"(started {start:+.2f}); {support:.0%} support vs {oppose:.0%} oppose."),
            confidence=confidence,
            key_drivers=["High-influence agents set the tone of the feed",
                         "Agreement clusters reinforced themselves through reposts"],
            turning_points=_biggest_swings(prompt) or ["No single round dominated"],
            scenarios=[
                Scenario(name="Yes", probability=p_yes, description="Supporters carry the outcome."),
                Scenario(name="No", probability=round(1 - p_yes, 2),
                         description="Opponents block or stall the outcome."),
            ],
            risks=["MOCK MODE: these agents follow simple word-matching rules, not a language model. "
                   "They do not understand the seed material. Use this run to try the app, not to make a "
                   "decision. Set ANTHROPIC_API_KEY for a real simulation."],
        )


def _biggest_swings(digest: str, n: int = 3) -> list[str]:
    rows = re.findall(r"^- (round \d+): stance ([-+0-9.]+) \(change ([-+0-9.]+)\)", digest, flags=re.M)
    rows.sort(key=lambda r: -abs(float(r[2])))
    return [f"{r}: mean stance moved {float(c):+.2f} to {float(st):+.2f}" for r, st, c in rows[:n] if float(c)]


def _field(text: str, key: str) -> str | None:
    match = re.search(rf"^{key}: ?(.*)$", text, flags=re.M)
    return match.group(1).strip() if match else None


def get_llm(mode: str | None = None) -> LLM:
    """``mode`` is "mock", "claude", or None (auto: Claude if credentials exist)."""

    mode = (mode or os.environ.get("SWARMSIGHT_LLM") or "auto").lower()
    if mode == "mock":
        return MockLLM()
    if mode == "claude":
        return ClaudeLLM()
    has_creds = any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"))
    return ClaudeLLM() if has_creds else MockLLM()
