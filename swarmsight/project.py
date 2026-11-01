"""A project ties one seed + question to its graph, world and report, on disk."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .graph import KnowledgeGraph, build_graph
from .llm import LLM
from .personas import build_personas
from .report import write_report
from .schemas import Injection, Report
from .simulation import World, run_simulation

Status = Literal["created", "building", "ready", "running", "reporting", "done", "error"]


class Project(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    title: str
    question: str
    seed: str
    status: Status = "created"
    message: str = ""
    max_agents: int = 24
    created_at: float = Field(default_factory=time.time)
    graph: KnowledgeGraph | None = None
    world: World | None = None
    report: Report | None = None

    def summary(self) -> dict:
        return {"id": self.id, "title": self.title, "question": self.question, "status": self.status,
                "message": self.message, "rounds": self.world.round if self.world else 0,
                "agents": len(self.world.agents) if self.world else 0, "created_at": self.created_at}


class ProjectStore:
    """JSON-file persistence plus background jobs, one at a time per project."""

    def __init__(self, root: Path, llm: LLM):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self._lock = threading.Lock()
        self._jobs: dict[str, threading.Thread] = {}
        self._cache: dict[str, Project] = {}

    # ---------------------------------------------------------- storage

    def _path(self, pid: str) -> Path:
        if not pid.isalnum():
            raise KeyError(pid)
        return self.root / f"{pid}.json"

    def save(self, project: Project) -> None:
        with self._lock:
            self._cache[project.id] = project
            tmp = self._path(project.id).with_suffix(".tmp")
            tmp.write_text(project.model_dump_json(), encoding="utf-8")
            tmp.replace(self._path(project.id))

    def get(self, pid: str) -> Project:
        with self._lock:
            if pid in self._cache:
                return self._cache[pid]
            path = self._path(pid)
            if not path.exists():
                raise KeyError(pid)
            project = Project.model_validate_json(path.read_text(encoding="utf-8"))
            self._cache[pid] = project
            return project

    def list(self) -> list[dict]:
        ids = {p.stem for p in self.root.glob("*.json")}
        return sorted((self.get(i).summary() for i in ids), key=lambda s: -s["created_at"])

    def create(self, title: str, question: str, seed: str, max_agents: int = 24) -> Project:
        if not seed.strip():
            raise ValueError("seed text is empty")
        if not question.strip():
            raise ValueError("question is empty")
        project = Project(title=title.strip() or question.strip()[:60], question=question.strip(),
                          seed=seed, max_agents=max_agents)
        self.save(project)
        return project

    # ------------------------------------------------------------- jobs

    def busy(self, pid: str) -> bool:
        job = self._jobs.get(pid)
        return job is not None and job.is_alive()

    def _start(self, project: Project, work) -> None:
        if self.busy(project.id):
            raise RuntimeError("a job is already running for this project")

        def runner():
            try:
                work()
            except Exception as exc:  # surface any failure to the UI
                project.status, project.message = "error", f"{type(exc).__name__}: {exc}"
                self.save(project)

        thread = threading.Thread(target=runner, daemon=True)
        self._jobs[project.id] = thread
        thread.start()

    def wait(self, pid: str, timeout: float | None = None) -> None:
        job = self._jobs.get(pid)
        if job:
            job.join(timeout)

    def build(self, project: Project) -> None:
        """Seed -> graph -> personas -> world."""

        def work():
            project.status, project.message = "building", "Extracting knowledge graph"
            self.save(project)
            project.graph = build_graph(project.seed, self.llm)
            project.message = f"Graph: {len(project.graph.entities)} entities. Creating personas"
            self.save(project)
            personas = build_personas(project.graph, project.question, self.llm, max_agents=project.max_agents)
            project.world = World.create(project.question, personas)
            project.status, project.message = "ready", f"{len(personas)} agents ready"
            self.save(project)

        self._start(project, work)

    def simulate(self, project: Project, rounds: int) -> None:
        if project.world is None:
            raise RuntimeError("build the project first")
        if not 1 <= rounds <= 100:
            raise ValueError("rounds must be between 1 and 100")

        def work():
            project.status, project.message = "running", f"Running {rounds} rounds"
            self.save(project)
            target = project.world.round + rounds

            def progress(stats):
                project.message = f"Round {stats.round}/{target}: mean stance {stats.mean_stance:+.2f}"
                self.save(project)

            run_simulation(project.world, self.llm, rounds, on_round=progress)
            project.status, project.message = "reporting", "Writing report"
            self.save(project)
            project.report = write_report(project.world, self.llm, project.graph)
            project.status, project.message = "done", project.report.headline
            self.save(project)

        self._start(project, work)

    def inject(self, project: Project, text: str, round_no: int | None = None) -> Injection:
        if project.world is None:
            raise RuntimeError("build the project first")
        if not text.strip():
            raise ValueError("event text is empty")
        round_no = round_no or project.world.round + 1
        if round_no <= project.world.round:
            raise ValueError(f"round {round_no} has already run")
        injection = Injection(round=round_no, text=text.strip())
        project.world.injections.append(injection)
        self.save(project)
        return injection


def export_json(project: Project) -> str:
    return json.dumps(project.model_dump(mode="json"), indent=2)
