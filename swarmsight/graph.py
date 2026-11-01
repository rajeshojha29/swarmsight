"""Seed text -> knowledge graph.

Seed material is split into overlapping chunks, each chunk is sent to the
LLM for entity/relation extraction, and the results are merged into a single
graph with entities de-duplicated by normalized name.
"""

from __future__ import annotations

import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from .llm import LLM
from .schemas import Entity, Extraction, Relation

EXTRACT_SYSTEM = """You build knowledge graphs from source documents for a social simulation.
Extract the entities that matter to the situation described and the relations between them.

Rules:
- kind is one of: person, organization, group, place, concept, event.
- Use "group" for collectives that act together (residents, unions, voters, fans).
- Use the most complete name that appears in the text. Do not invent entities.
- summary: one sentence on who/what the entity is and its position in this situation.
- relations connect entity names exactly as you wrote them; label is a short verb phrase.
- evidence quotes or closely paraphrases the supporting sentence."""


def chunk_text(text: str, size: int = 3000, overlap: int = 300) -> list[str]:
    text = re.sub(r"\s+\n", "\n", text).strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        # Prefer to cut at a paragraph or sentence boundary.
        cut = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
        if end < len(text) and cut > start + size // 2:
            end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def normalize(name: str) -> str:
    name = re.sub(r"^(the|a|an)\s+", "", name.strip(), flags=re.I)
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


class KnowledgeGraph(BaseModel):
    entities: dict[str, Entity] = Field(default_factory=dict)  # key: normalized name
    relations: list[Relation] = Field(default_factory=list)

    def add(self, extraction: Extraction) -> None:
        for entity in extraction.entities:
            key = normalize(entity.name)
            if not key:
                continue
            existing = self.entities.get(key)
            if existing is None:
                self.entities[key] = entity.model_copy()
            elif len(entity.summary) > len(existing.summary):
                existing.summary = entity.summary
        seen = {(normalize(r.source), normalize(r.target), r.label) for r in self.relations}
        for rel in extraction.relations:
            s, t = normalize(rel.source), normalize(rel.target)
            if s in self.entities and t in self.entities and s != t and (s, t, rel.label) not in seen:
                seen.add((s, t, rel.label))
                self.relations.append(Relation(source=self.entities[s].name, target=self.entities[t].name,
                                               label=rel.label, evidence=rel.evidence))

    def consolidate(self) -> None:
        """Merge name variants of the same entity, e.g. "Transit Authority" into
        "Harborview Transit Authority": same kind, and the shorter name (two or
        more words) is the tail of the longer one."""
        keys = sorted(self.entities, key=lambda k: (len(k.split()), k))
        alias: dict[str, str] = {}
        for short in keys:
            if len(short.split()) < 2:
                continue
            kind = self.entities[short].kind
            longer = [k for k in keys if k != short and k.endswith(" " + short)
                      and self.entities[k].kind == kind and k not in alias]
            if len(longer) == 1:  # only merge when the match is unambiguous
                alias[short] = longer[0]
        if not alias:
            return
        names = {k: self.entities[v].name for k, v in alias.items()}
        for short in alias:
            del self.entities[short]
        merged, seen = [], set()
        for rel in self.relations:
            s = names.get(normalize(rel.source), rel.source)
            t = names.get(normalize(rel.target), rel.target)
            key = (normalize(s), normalize(t), rel.label)
            if normalize(s) != normalize(t) and key not in seen:
                seen.add(key)
                merged.append(rel.model_copy(update={"source": s, "target": t}))
        self.relations = merged

    def degree(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for rel in self.relations:
            counts[normalize(rel.source)] += 1
            counts[normalize(rel.target)] += 1
        return counts

    def neighbors(self, name: str) -> list[tuple[str, str]]:
        key = normalize(name)
        out = []
        for rel in self.relations:
            if normalize(rel.source) == key:
                out.append((rel.label, rel.target))
            elif normalize(rel.target) == key:
                out.append((f"is {rel.label} by" if not rel.label.startswith("is ") else rel.label, rel.source))
        return out

    def context_for(self, name: str, limit: int = 8) -> str:
        entity = self.entities.get(normalize(name))
        lines = [entity.summary] if entity and entity.summary else []
        lines += [f"{name} {label} {other}." for label, other in self.neighbors(name)[:limit]]
        return " ".join(lines)

    def search(self, query: str, limit: int = 10) -> list[Entity]:
        terms = set(normalize(query).split())
        scored = []
        for key, entity in self.entities.items():
            hay = set(key.split()) | set(normalize(entity.summary).split())
            score = len(terms & hay)
            if score:
                scored.append((score, entity.name, entity))
        return [e for _, _, e in sorted(scored, key=lambda x: (-x[0], x[1]))[:limit]]

    def to_view(self) -> dict:
        """Nodes/links shape for the web UI."""
        deg = self.degree()
        return {
            "nodes": [{"id": e.name, "kind": e.kind, "summary": e.summary, "degree": deg.get(k, 0)}
                      for k, e in self.entities.items()],
            "links": [{"source": r.source, "target": r.target, "label": r.label} for r in self.relations],
        }


def build_graph(text: str, llm: LLM, *, max_workers: int = 4) -> KnowledgeGraph:
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("seed text is empty")

    def extract(chunk: str) -> Extraction:
        return llm.structured(EXTRACT_SYSTEM, f"TEXT:\n{chunk}", Extraction, task="extract")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(extract, chunks))

    graph = KnowledgeGraph()
    for result in results:  # merge in chunk order so output is deterministic
        graph.add(result)
    graph.consolidate()
    return graph
