import pytest

from conftest import QUESTION
from swarmsight.graph import build_graph
from swarmsight.llm import LLMError, MockLLM
from swarmsight.personas import build_personas, make_id
from swarmsight.schemas import AgentDecision, Injection, Persona
from swarmsight.simulation import World, run_simulation


def make_world(seed, llm, max_agents=12):
    graph = build_graph(seed, llm)
    return World.create(QUESTION, build_personas(graph, QUESTION, llm, max_agents=max_agents))


def test_personas_are_actors_with_unique_ids(seed, llm):
    personas = build_personas(build_graph(seed, llm), QUESTION, llm, max_agents=10)
    assert 2 <= len(personas) <= 10
    assert len({p.id for p in personas}) == len(personas)
    assert all(p.kind in {"person", "organization", "group"} for p in personas)
    assert all(-1 <= p.stance <= 1 for p in personas)


def test_make_id_dedupes():
    taken = set()
    assert make_id("Elena Ruiz", taken) == "elena-ruiz"
    assert make_id("Elena Ruiz", taken) == "elena-ruiz-2"


def test_world_follow_graph(seed, llm):
    world = make_world(seed, llm)
    for aid, agent in world.agents.items():
        assert aid not in agent.following
        assert set(agent.following) <= set(world.agents)
    assert world.stats[0].round == 0


def test_run_is_reproducible(seed, llm):
    a = run_simulation(make_world(seed, llm), llm, 4)
    b = run_simulation(make_world(seed, MockLLM()), MockLLM(), 4)
    assert a.round == 4 and len(a.stats) == 5
    assert [p.model_dump() for p in a.posts] == [p.model_dump() for p in b.posts]
    assert a.posts, "agents should have posted something"
    assert any(p.kind == "reply" for p in a.posts)


def test_injection_becomes_event_and_moves_stance(seed, llm):
    base = run_simulation(make_world(seed, llm), llm, 3)
    shocked = make_world(seed, llm)
    shocked.injections.append(Injection(round=2, text="Every group now welcomes and supports the plan"))
    run_simulation(shocked, llm, 3)
    events = [p for p in shocked.posts if p.kind == "event"]
    assert len(events) == 1 and events[0].round == 2
    assert shocked.stats[2].mean_stance > base.stats[2].mean_stance


class ScriptedLLM:
    """Returns a fixed decision; fails for one agent."""

    name = "scripted"

    def __init__(self, decision, fail_for=None):
        self.decision, self.fail_for = decision, fail_for

    def structured(self, system, prompt, schema, *, task):
        if self.fail_for and f"AGENT_ID: {self.fail_for}" in prompt:
            raise LLMError("boom")
        return self.decision

    def text(self, system, messages, *, task):
        return "ok"


def personas(n=3):
    return [Persona(id=f"a{i}", name=f"Agent {i}", kind="person", bio="", stance=0.0) for i in range(n)]


def test_missing_target_falls_back_and_errors_are_recorded():
    world = World.create("Q?", personas())
    llm = ScriptedLLM(AgentDecision(action="reply", target_post_id="p999", content="hello", stance=0.5),
                      fail_for="a1")
    stats = world.run_round(llm)
    assert stats.posts == 2 and stats.idle == 1  # bad reply target -> plain post
    assert all(p.parent_id is None for p in world.posts)
    assert world.errors and "a1" in world.errors[0]
    assert world.agents["a0"].stance == 0.5 and world.agents["a1"].stance == 0.0


def test_likes_are_counted_once():
    world = World.create("Q?", personas(2))
    world.run_round(ScriptedLLM(AgentDecision(action="post", content="first", stance=0.1)))
    target = world.posts[0]
    like = ScriptedLLM(AgentDecision(action="like", target_post_id=target.id, stance=0.1))
    world.run_round(like)
    world.run_round(like)
    assert len(target.likes) == 2  # both agents, each once
    assert len(set(target.likes)) == len(target.likes)


def test_world_needs_actors(llm):
    from swarmsight.graph import KnowledgeGraph
    with pytest.raises(ValueError):
        build_personas(KnowledgeGraph(), QUESTION, llm)
