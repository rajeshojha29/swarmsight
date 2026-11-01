"""Command line entry point: ``swarmsight run`` and ``swarmsight serve``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .graph import build_graph
from .llm import get_llm
from .personas import build_personas
from .report import write_report
from .schemas import Injection
from .simulation import World, run_simulation


def _run(args: argparse.Namespace) -> int:
    llm = get_llm("mock" if args.mock else args.llm)
    seed = Path(args.seed).read_text(encoding="utf-8")
    print(f"LLM: {llm.name}")

    graph = build_graph(seed, llm)
    print(f"Graph: {len(graph.entities)} entities, {len(graph.relations)} relations")
    personas = build_personas(graph, args.question, llm, max_agents=args.agents)
    print(f"Agents: {', '.join(p.name for p in personas)}")

    world = World.create(args.question, personas)
    for spec in args.inject or []:
        round_no, _, text = spec.partition(":")
        world.injections.append(Injection(round=int(round_no), text=text.strip()))

    run_simulation(world, llm, args.rounds, on_round=lambda s: print(
        f"  round {s.round}: stance {s.mean_stance:+.2f}  support {s.support_share:.0%}  "
        f"oppose {s.oppose_share:.0%}  ({s.posts} posts, {s.replies} replies)"))
    for err in world.errors:
        print(f"  warning: {err}", file=sys.stderr)

    report = write_report(world, llm, graph)
    if args.out:
        Path(args.out).write_text(report.markdown, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print()
        print(report.markdown)
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    app = create_app(get_llm("mock" if args.mock else args.llm), args.data)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swarmsight", description="Swarm-simulation forecasting engine")
    sub = parser.add_subparsers(dest="command", required=True)

    def llm_flags(p):
        p.add_argument("--llm", choices=["auto", "claude", "mock"], default="auto",
                       help="auto uses Claude when ANTHROPIC_API_KEY is set, otherwise mock")
        p.add_argument("--mock", action="store_true", help="shorthand for --llm mock")

    run = sub.add_parser("run", help="run a full simulation from a seed file")
    run.add_argument("seed", help="path to a text/markdown seed file")
    run.add_argument("-q", "--question", required=True, help="what you want to predict")
    run.add_argument("-r", "--rounds", type=int, default=5)
    run.add_argument("-a", "--agents", type=int, default=24, help="maximum number of agents")
    run.add_argument("-i", "--inject", action="append", metavar="ROUND:TEXT",
                     help='inject an event, e.g. "3:The regulator delays the vote"')
    run.add_argument("-o", "--out", help="write the markdown report here")
    llm_flags(run)
    run.set_defaults(func=_run)

    serve = sub.add_parser("serve", help="start the web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--data", default="data/projects")
    llm_flags(serve)
    serve.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
