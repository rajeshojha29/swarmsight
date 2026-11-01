"""HTTP API and web UI."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .llm import LLM, get_llm
from .project import ProjectStore
from .report import chat_with_agent, chat_with_analyst

WEB = Path(__file__).parent / "web"


class CreateBody(BaseModel):
    title: str = ""
    question: str
    seed: str
    max_agents: int = Field(24, ge=2, le=200)


class RunBody(BaseModel):
    rounds: int = Field(5, ge=1, le=100)


class InjectBody(BaseModel):
    text: str
    round: int | None = None


class ChatBody(BaseModel):
    agent_id: str | None = None  # None -> chat with the report analyst
    messages: list[dict]


def create_app(llm: LLM | None = None, data_dir: str | Path | None = None) -> FastAPI:
    llm = llm or get_llm()
    store = ProjectStore(Path(data_dir or os.environ.get("SWARMSIGHT_DATA", "data/projects")), llm)
    app = FastAPI(title="SwarmSight")
    app.state.store = store

    def load(pid: str):
        try:
            return store.get(pid)
        except KeyError:
            raise HTTPException(404, "project not found")

    def guard(fn):
        try:
            return fn()
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/health")
    def health():
        return {"ok": True, "llm": llm.name}

    @app.get("/api/projects")
    def list_projects():
        return store.list()

    @app.post("/api/projects", status_code=201)
    def create(body: CreateBody):
        project = guard(lambda: store.create(body.title, body.question, body.seed, body.max_agents))
        store.build(project)
        return project.summary()

    @app.get("/api/projects/{pid}")
    def get_project(pid: str):
        p = load(pid)
        world = p.world
        return {
            **p.summary(),
            "busy": store.busy(pid),
            "graph": p.graph.to_view() if p.graph else None,
            "agents": [{"id": a.persona.id, "name": a.persona.name, "kind": a.persona.kind,
                        "bio": a.persona.bio, "traits": a.persona.traits, "influence": a.persona.influence,
                        "start_stance": a.persona.stance, "stance": a.stance}
                       for a in world.agents.values()] if world else [],
            "posts": [x.model_dump() | {"engagement": x.engagement} for x in world.posts[-300:]] if world else [],
            "stats": [s.model_dump() for s in world.stats] if world else [],
            "injections": [i.model_dump() for i in world.injections] if world else [],
            "errors": world.errors[-20:] if world else [],
            "report": p.report.model_dump() if p.report else None,
        }

    @app.post("/api/projects/{pid}/run", status_code=202)
    def run(pid: str, body: RunBody):
        p = load(pid)
        guard(lambda: store.simulate(p, body.rounds))
        return p.summary()

    @app.post("/api/projects/{pid}/inject", status_code=201)
    def inject(pid: str, body: InjectBody):
        p = load(pid)
        if store.busy(pid):
            raise HTTPException(409, "wait for the current job to finish")
        return guard(lambda: store.inject(p, body.text, body.round)).model_dump()

    @app.post("/api/projects/{pid}/chat")
    def chat(pid: str, body: ChatBody):
        p = load(pid)
        if not p.world:
            raise HTTPException(409, "build the project first")
        if not body.messages or body.messages[-1].get("role") != "user":
            raise HTTPException(400, "last message must be from the user")
        if body.agent_id:
            try:
                return {"reply": chat_with_agent(p.world, body.agent_id, body.messages, llm)}
            except KeyError:
                raise HTTPException(404, "agent not found")
        if not p.report:
            raise HTTPException(409, "run the simulation to get a report first")
        return {"reply": chat_with_analyst(p.world, p.report, body.messages, llm, p.graph)}

    @app.get("/api/projects/{pid}/report.md", response_class=PlainTextResponse)
    def report_md(pid: str):
        p = load(pid)
        if not p.report:
            raise HTTPException(404, "no report yet")
        return p.report.markdown

    return app
