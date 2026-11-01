from swarmsight.graph import KnowledgeGraph, build_graph, chunk_text, normalize
from swarmsight.schemas import Entity, Extraction, Relation


def test_chunk_text_short_and_long():
    assert chunk_text("") == []
    assert chunk_text("one sentence.") == ["one sentence."]
    text = " ".join(f"Sentence number {i} is here." for i in range(600))
    chunks = chunk_text(text, size=1000, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    # Every sentence survives chunking.
    assert all(f"number {i} " in " ".join(chunks) for i in range(600))


def test_normalize():
    assert normalize("The Harborview Courier") == "harborview courier"
    assert normalize("  Fair-Fares NOW! ") == "fair fares now"


def test_add_dedupes_and_drops_dangling_relations():
    g = KnowledgeGraph()
    g.add(Extraction(
        entities=[Entity(name="Elena Ruiz", kind="person"), Entity(name="City Council", kind="organization")],
        relations=[Relation(source="Elena Ruiz", target="City Council", label="lobbies"),
                   Relation(source="Elena Ruiz", target="Nobody", label="knows")]))
    g.add(Extraction(
        entities=[Entity(name="the City Council", kind="organization", summary="Votes next month.")],
        relations=[Relation(source="elena ruiz", target="the city council", label="lobbies")]))
    assert len(g.entities) == 2
    assert g.entities["city council"].summary == "Votes next month."
    assert len(g.relations) == 1


def test_consolidate_merges_name_variants():
    g = KnowledgeGraph()
    g.add(Extraction(
        entities=[Entity(name="Harborview Transit Authority", kind="organization"),
                  Entity(name="Transit Authority", kind="organization"),
                  Entity(name="Dana Okafor", kind="person")],
        relations=[Relation(source="Dana Okafor", target="Transit Authority", label="leads")]))
    g.consolidate()
    assert "transit authority" not in g.entities
    assert g.relations[0].target == "Harborview Transit Authority"


def test_build_graph_from_example(seed, llm):
    g = build_graph(seed, llm)
    names = {e.name for e in g.entities.values()}
    assert {"Elena Ruiz", "Marcus Webb", "Rideshare Drivers Union"} <= names
    assert g.relations
    view = g.to_view()
    assert len(view["nodes"]) == len(g.entities)
    assert "Elena Ruiz" in g.context_for("Elena Ruiz")
    assert g.search("transit")
