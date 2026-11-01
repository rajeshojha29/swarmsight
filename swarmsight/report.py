"""Post-simulation analysis: the prediction report and chat with agents."""

from __future__ import annotations

from .graph import KnowledgeGraph
from .llm import LLM
from .schemas import Report
from .simulation import World

REPORT_SYSTEM = """You are a forecasting analyst. You are given the full record of a multi-agent
social simulation run to explore a question: the starting knowledge graph, the actors, how
their stances moved round by round, injected events, and the most influential posts.

Write a prediction grounded in that evidence. Cite specific actors, rounds and posts.
confidence (0-1) reflects how decisive and stable the simulated outcome was. Scenario
probabilities should sum to roughly 1. Be explicit about the limits of a simulation."""

AGENT_CHAT_SYSTEM = """You are {name}, an actor from a social simulation, being interviewed after it ended.
Answer in first person, in your own voice ({voice}), consistent with your bio, goals and memory.
If asked about something you did not experience, say so.

NAME: {name}
BIO: {bio}
TRAITS: {traits}
GOALS: {goals}
STANCE: {stance:+.2f}
LAST_MEMORY: {last_memory}

MEMORY:
{memory}"""

ANALYST_CHAT_SYSTEM = """You are the analyst who wrote the report below about a social simulation.
Answer follow-up questions using the report and the simulation evidence. Be concise and say when
the simulation does not support an answer.

{evidence}

REPORT:
{report}"""


def evidence_digest(world: World, graph: KnowledgeGraph | None = None, top_posts: int = 12) -> str:
    first, last = world.stats[0], world.stats[-1]
    lines = [
        f"QUESTION: {world.question}",
        f"ROUNDS: {world.round}",
        f"AGENTS: {len(world.agents)}",
        f"START_STANCE: {first.mean_stance:+.3f}",
        f"END_STANCE: {last.mean_stance:+.3f}",
        f"END_SUPPORT: {last.support_share:.3f}",
        f"END_OPPOSE: {last.oppose_share:.3f}",
        "",
        "ROUND BY ROUND (mean stance, support/oppose share, activity):",
    ]
    prev = first.mean_stance
    for s in world.stats[1:]:
        lines.append(f"- round {s.round}: stance {s.mean_stance:+.3f} (change {s.mean_stance - prev:+.3f}), "
                     f"support {s.support_share:.0%}, oppose {s.oppose_share:.0%}, "
                     f"{s.posts} posts / {s.replies} replies / {s.reposts} reposts / {s.likes} likes")
        prev = s.mean_stance
    if world.injections:
        lines += ["", "INJECTED EVENTS:"] + [f"- round {i.round}: {i.text}" for i in world.injections]

    lines += ["", "ACTORS (start -> end stance, influence):"]
    for a in sorted(world.agents.values(), key=lambda a: -a.persona.influence):
        lines.append(f"- {a.persona.name} [{a.persona.kind}]: {a.persona.stance:+.2f} -> {a.stance:+.2f}, "
                     f"influence {a.persona.influence:.2f}")

    ranked = sorted((p for p in world.posts if p.kind != "event"), key=lambda p: (-p.engagement, p.id))
    lines += ["", "MOST ENGAGED POSTS:"]
    lines += [f"- [{p.id}] round {p.round} {p.author_name} ({p.engagement} engagement): {p.content}"
              for p in ranked[:top_posts]] or ["- none"]

    if graph is not None:
        lines += ["", f"KNOWLEDGE GRAPH: {len(graph.entities)} entities, {len(graph.relations)} relations"]
        lines += [f"- {r.source} {r.label} {r.target}" for r in graph.relations[:20]]
    return "\n".join(lines)


def to_markdown(report: Report, world: World) -> str:
    out = [f"# {report.headline}", "", f"**Question:** {world.question}", "",
           f"**Prediction:** {report.prediction}", "",
           f"**Confidence:** {report.confidence:.0%}", "", "## Key drivers"]
    out += [f"- {d}" for d in report.key_drivers]
    out += ["", "## Turning points"] + [f"- {t}" for t in report.turning_points]
    out += ["", "## Scenarios", "", "| Scenario | Probability | Description |", "|---|---|---|"]
    out += [f"| {s.name} | {s.probability:.0%} | {s.description} |" for s in report.scenarios]
    out += ["", "## Risks and limits"] + [f"- {r}" for r in report.risks]
    out += ["", "## Stance by round", "", "| Round | Mean stance | Support | Oppose |", "|---|---|---|---|"]
    out += [f"| {s.round} | {s.mean_stance:+.2f} | {s.support_share:.0%} | {s.oppose_share:.0%} |"
            for s in world.stats]
    return "\n".join(out) + "\n"


def write_report(world: World, llm: LLM, graph: KnowledgeGraph | None = None) -> Report:
    if world.round == 0:
        raise ValueError("run at least one round before writing a report")
    report = llm.structured(REPORT_SYSTEM, evidence_digest(world, graph), Report, task="report")
    report.markdown = to_markdown(report, world)
    return report


def chat_with_agent(world: World, agent_id: str, history: list[dict], llm: LLM) -> str:
    agent = world.agents.get(agent_id)
    if agent is None:
        raise KeyError(agent_id)
    p = agent.persona
    system = AGENT_CHAT_SYSTEM.format(
        name=p.name, voice=p.voice, bio=p.bio, traits=", ".join(p.traits), goals="; ".join(p.goals),
        stance=agent.stance, last_memory=agent.memory[-1] if agent.memory else "",
        memory="\n".join(f"- {m}" for m in agent.memory) or "- (nothing)")
    return llm.text(system, history, task="chat")


def chat_with_analyst(world: World, report: Report, history: list[dict], llm: LLM,
                      graph: KnowledgeGraph | None = None) -> str:
    system = ANALYST_CHAT_SYSTEM.format(evidence=evidence_digest(world, graph), report=report.markdown)
    return llm.text(system, history, task="chat")
