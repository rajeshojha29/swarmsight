import pytest

from conftest import QUESTION
from swarmsight.graph import build_graph
from swarmsight.personas import build_personas
from swarmsight.report import chat_with_agent, chat_with_analyst, evidence_digest, write_report
from swarmsight.simulation import World, run_simulation


@pytest.fixture
def finished(seed, llm):
    graph = build_graph(seed, llm)
    world = World.create(QUESTION, build_personas(graph, QUESTION, llm, max_agents=10))
    return graph, run_simulation(world, llm, 3)


def test_report_needs_a_run(seed, llm):
    graph = build_graph(seed, llm)
    world = World.create(QUESTION, build_personas(graph, QUESTION, llm, max_agents=5))
    with pytest.raises(ValueError):
        write_report(world, llm)


def test_report_markdown(finished, llm):
    graph, world = finished
    report = write_report(world, llm, graph)
    assert 0 <= report.confidence <= 1
    assert report.scenarios
    assert QUESTION in report.markdown
    assert "| 3 |" in report.markdown  # stance table covers every round


def test_digest_lists_rounds_and_actors(finished):
    graph, world = finished
    digest = evidence_digest(world, graph)
    for r in (1, 2, 3):
        assert f"- round {r}:" in digest
    assert all(a.persona.name in digest for a in world.agents.values())


def test_chat(finished, llm):
    graph, world = finished
    agent_id = next(iter(world.agents))
    reply = chat_with_agent(world, agent_id, [{"role": "user", "content": "Why?"}], llm)
    assert world.agents[agent_id].persona.name in reply
    with pytest.raises(KeyError):
        chat_with_agent(world, "nobody", [{"role": "user", "content": "hi"}], llm)
    report = write_report(world, llm, graph)
    assert chat_with_analyst(world, report, [{"role": "user", "content": "Summarise"}], llm, graph)
