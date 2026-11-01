"""Knowledge-graph entities -> simulated agents with personas."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from .graph import KnowledgeGraph, normalize
from .llm import LLM
from .schemas import ACTOR_KINDS, Entity, Persona, PersonaDraft

PERSONA_SYSTEM = """You create realistic personas for actors in a social simulation.
Each persona is grounded in what the source material says about the actor.

stance is the actor's starting position on the prediction question, from -1.0 (strongly
against the outcome) to +1.0 (strongly for it); use values near 0 when the material is
silent. influence (0-1) is how much attention their posts get; organizations and public
figures are usually higher. activity (0-1) is how often they post. voice is a few words
describing how they write. Keep bio to two sentences."""


def select_actors(graph: KnowledgeGraph, max_agents: int) -> list[Entity]:
    """The best-connected actor entities, most central first."""
    degree = graph.degree()
    actors = [(normalize(e.name), e) for e in graph.entities.values() if e.kind in ACTOR_KINDS]
    actors.sort(key=lambda item: (-degree.get(item[0], 0), item[1].name))
    return [e for _, e in actors[:max_agents]]


def make_id(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32] or "agent"
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}-{n}", n + 1
    taken.add(candidate)
    return candidate


def build_personas(graph: KnowledgeGraph, question: str, llm: LLM, *,
                   max_agents: int = 24, max_workers: int = 4) -> list[Persona]:
    actors = select_actors(graph, max_agents)
    if not actors:
        raise ValueError("no people, organizations or groups found in the seed material")

    def draft(entity: Entity) -> PersonaDraft:
        prompt = (f"QUESTION: {question}\nNAME: {entity.name}\nKIND: {entity.kind}\n"
                  f"CONTEXT: {graph.context_for(entity.name)}")
        return llm.structured(PERSONA_SYSTEM, prompt, PersonaDraft, task="persona")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        drafts = list(pool.map(draft, actors))

    taken: set[str] = set()
    return [Persona(id=make_id(e.name, taken), name=e.name, kind=e.kind, **d.model_dump())
            for e, d in zip(actors, drafts)]
