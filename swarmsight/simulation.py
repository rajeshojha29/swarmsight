"""Multi-agent social simulation.

Each round, every agent reads a personalised feed, consults its own memory,
and decides to post, reply, repost, like, or stay idle. Decisions for a round
are made in parallel against the same snapshot of the world, then applied in
a fixed order, so a run is reproducible for a given LLM.

An observer can inject events ("the regulator delays the vote") at any round;
every agent sees them in that round's feed.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from pydantic import BaseModel, Field

from .llm import LLM, LLMError
from .schemas import AgentDecision, Injection, Persona, Post, RoundStats

ACT_SYSTEM = """You are role-playing one actor in a social media simulation about a real-world question.
Stay in character: write in the persona's voice, pursue their goals, and react to what is in
the feed and to any WORLD_EVENT. Be specific and plausible; no meta-commentary.

Choose exactly one action:
- post: share a new message (content required, under 280 characters)
- reply: respond to a feed item (target_post_id and content required)
- repost: amplify a feed item (target_post_id required)
- like: endorse a feed item (target_post_id required)
- idle: do nothing this round

stance is your position on the question AFTER this round, from -1.0 (strongly against) to
+1.0 (strongly for). Shift it only as much as this persona plausibly would.
reflection is one private sentence about what you noticed or decided."""

FEED_SIZE = 8
MEMORY_SIZE = 12


class AgentState(BaseModel):
    persona: Persona
    stance: float
    memory: list[str] = Field(default_factory=list)
    following: list[str] = Field(default_factory=list)

    def remember(self, note: str) -> None:
        self.memory.append(note)
        del self.memory[:-MEMORY_SIZE]


class World(BaseModel):
    question: str
    agents: dict[str, AgentState]
    posts: list[Post] = Field(default_factory=list)
    injections: list[Injection] = Field(default_factory=list)
    stats: list[RoundStats] = Field(default_factory=list)
    round: int = 0
    errors: list[str] = Field(default_factory=list)

    @classmethod
    def create(cls, question: str, personas: list[Persona]) -> "World":
        agents = {p.id: AgentState(persona=p, stance=p.stance) for p in personas}
        # Everyone follows the most influential actors, plus a few peers.
        by_influence = sorted(personas, key=lambda p: (-p.influence, p.id))
        hubs = [p.id for p in by_influence[: max(2, len(personas) // 5)]]
        ids = [p.id for p in personas]
        for i, p in enumerate(personas):
            peers = [ids[(i + k) % len(ids)] for k in (1, 2, 3)]
            agents[p.id].following = sorted({*hubs, *peers} - {p.id})
        world = cls(question=question, agents=agents)
        world.stats.append(world._stats(0, []))
        return world

    # ------------------------------------------------------------- feed

    def post_by_id(self, post_id: str | None) -> Post | None:
        return next((p for p in self.posts if p.id == post_id), None)

    def feed_for(self, agent: AgentState, round_no: int) -> list[Post]:
        followed = set(agent.following)

        def score(post: Post) -> float:
            author = self.agents.get(post.author_id)
            influence = author.persona.influence if author else 1.0
            recency = 1.0 / (1 + round_no - post.round)
            social = 1.5 if post.author_id in followed else 1.0
            return (1 + post.engagement) * (0.5 + influence) * recency * social

        candidates = [p for p in self.posts if p.author_id != agent.persona.id and p.kind != "event"]
        candidates.sort(key=lambda p: (-score(p), p.id))
        return candidates[:FEED_SIZE]

    # ------------------------------------------------------------- round

    def prompt_for(self, agent: AgentState, round_no: int) -> str:
        p = agent.persona
        events = [i.text for i in self.injections if i.round == round_no]
        feed = self.feed_for(agent, round_no)
        feed_lines = [f"[{post.id}] {post.author_name} (stance {post.stance:+.2f}, engagement {post.engagement}): "
                      f"{post.content}" for post in feed] or ["(the feed is empty)"]
        memory = [f"- {m}" for m in agent.memory] or ["- (nothing yet)"]
        return "\n".join([
            f"QUESTION: {self.question}",
            f"AGENT_ID: {p.id}",
            f"NAME: {p.name}",
            f"KIND: {p.kind}",
            f"BIO: {p.bio}",
            f"TRAITS: {', '.join(p.traits)}",
            f"GOALS: {'; '.join(p.goals)}",
            f"VOICE: {p.voice}",
            f"ROUND: {round_no}",
            f"STANCE: {agent.stance:+.3f}",
            f"ACTIVITY: {p.activity}",
            *( [f"WORLD_EVENT: {' '.join(events)}"] if events else [] ),
            "",
            "YOUR MEMORY:",
            *memory,
            "",
            "FEED (author leaning shown as stance):",
            *feed_lines,
            "",
            f"Your activity level is {p.activity:.2f}: at lower levels, idle is often the realistic choice.",
        ])

    def run_round(self, llm: LLM, *, max_workers: int = 6) -> RoundStats:
        round_no = self.round + 1
        for inj in self.injections:
            if inj.round == round_no:
                self.posts.append(Post(id=f"p{len(self.posts) + 1}", round=round_no, author_id="world",
                                       author_name="World event", kind="event", content=inj.text))

        order = sorted(self.agents)
        prompts = {aid: self.prompt_for(self.agents[aid], round_no) for aid in order}

        def decide(aid: str) -> AgentDecision | None:
            try:
                return llm.structured(ACT_SYSTEM, prompts[aid], AgentDecision, task="act")
            except (LLMError, ValueError) as exc:
                self.errors.append(f"round {round_no} {aid}: {exc}")
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            decisions = dict(zip(order, pool.map(decide, order)))

        actions: list[str] = []
        for aid in order:
            decision = decisions[aid]
            if decision is None:
                actions.append("idle")
                continue
            actions.append(self._apply(self.agents[aid], decision, round_no))

        self.round = round_no
        stats = self._stats(round_no, actions)
        self.stats.append(stats)
        return stats

    def _apply(self, agent: AgentState, d: AgentDecision, round_no: int) -> str:
        p = agent.persona
        agent.stance = d.stance
        target = self.post_by_id(d.target_post_id)
        action = d.action
        # Fall back to a plain post/idle if the model referenced a missing post.
        if action in ("reply", "repost", "like") and target is None:
            action = "post" if d.content else "idle"

        if action in ("post", "reply") and d.content.strip():
            self.posts.append(Post(
                id=f"p{len(self.posts) + 1}", round=round_no, author_id=p.id, author_name=p.name,
                kind=action, content=d.content.strip()[:500], stance=d.stance,
                parent_id=target.id if action == "reply" and target else None))
            agent.remember(f"Round {round_no}: I {'replied to ' + target.author_name if action == 'reply' and target else 'posted'}: {d.content.strip()[:140]}")
        elif action == "repost" and target:
            if p.id not in target.reposts:
                target.reposts.append(p.id)
            agent.remember(f"Round {round_no}: I reposted {target.author_name}: {target.content[:100]}")
        elif action == "like" and target:
            if p.id not in target.likes:
                target.likes.append(p.id)
            agent.remember(f"Round {round_no}: I liked {target.author_name}'s post.")
        else:
            action = "idle"
        if d.reflection:
            agent.remember(f"Round {round_no} thought: {d.reflection[:160]}")
        return action

    def _stats(self, round_no: int, actions: list[str]) -> RoundStats:
        stances = [a.stance for a in self.agents.values()]
        n = len(stances) or 1
        return RoundStats(
            round=round_no,
            posts=actions.count("post"), replies=actions.count("reply"),
            reposts=actions.count("repost"), likes=actions.count("like"), idle=actions.count("idle"),
            mean_stance=round(sum(stances) / n, 4),
            support_share=round(sum(s > 0.2 for s in stances) / n, 4),
            oppose_share=round(sum(s < -0.2 for s in stances) / n, 4),
            neutral_share=round(sum(-0.2 <= s <= 0.2 for s in stances) / n, 4),
        )


def run_simulation(world: World, llm: LLM, rounds: int, *,
                   on_round: Callable[[RoundStats], None] | None = None,
                   stop: threading.Event | None = None) -> World:
    for _ in range(rounds):
        if stop is not None and stop.is_set():
            break
        stats = world.run_round(llm)
        if on_round:
            on_round(stats)
    return world
