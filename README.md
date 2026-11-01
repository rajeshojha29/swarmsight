# SwarmSight

**Rehearse the future with a swarm of AI agents.**

Give SwarmSight some seed material (news articles, a policy draft, a market report, even a story) and a question you want to predict. It builds a small world from that material: a knowledge graph of who and what is involved, and an AI persona for every person, organization and group in it. It then lets those agents argue it out on a simulated social feed, round after round. When the run ends, a report agent writes up a prediction from what happened. You can also interview any agent about why they acted as they did.

You can step in from a "god's-eye view" at any time. Inject an event ("the regulator delays the vote") and watch how the swarm reacts.

> SwarmSight is an independent project inspired by the swarm-simulation idea behind [MiroFish](https://github.com/666ghj/MiroFish). It is written from scratch and shares no code with it.

## How it works

```
seed text ──► knowledge graph ──► personas ──► social simulation ──► report ──► chat
             (entities +          (bio, goals,   (feed, memory,        (prediction,   (any agent
              relations)           stance,        posts, replies,       scenarios,     or the
                                   influence)     reposts, likes,       confidence)    analyst)
                                                  injected events)
```

1. **Graph building.** The seed is chunked, and Claude extracts entities and relations from each chunk with structured outputs. Results are merged, and duplicate names ("Transit Authority" / "Harborview Transit Authority") are consolidated.
2. **Personas.** The best-connected people, organizations and groups become agents. Each gets a bio, traits, goals, a writing voice, a starting **stance** on the question (−1 to +1), an **influence** score and an **activity** level, all grounded in the graph.
3. **Simulation.** Every round, each agent reads a personalised feed (ranked by engagement, the author's influence, recency and who they follow), consults its own memory, and decides to post, reply, repost, like, or stay idle. It also updates its stance. Decisions for a round are made in parallel against the same snapshot, then applied in a fixed order.
4. **Report.** A report agent reads the full record: stance trajectory, injected events, the most engaged posts and how each actor moved. It writes a prediction with confidence, key drivers, turning points, scenarios and risks.
5. **Deep interaction.** Chat with any agent (it answers from its persona and memory), or ask the analyst follow-up questions about the report.

## Quick start

Requires Python 3.10+.

```bash
pip install -e ".[dev]"

# Offline demo: no API key needed, deterministic heuristic agents
swarmsight run examples/harborview_congestion.md \
  -q "Will the Harborview City Council approve the congestion charge?" \
  -r 6 -i "4:Councillor Marcus Webb wins a low-income discount and now backs the plan" --mock

# Web UI
swarmsight serve --mock          # http://127.0.0.1:8000
```

### Using Claude

Set an Anthropic API key (from [console.anthropic.com](https://console.anthropic.com)) and drop `--mock`.

macOS / Linux:

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key
swarmsight serve
```

Windows PowerShell:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-your-key"   # this window only
swarmsight serve
```

Windows Command Prompt: `set ANTHROPIC_API_KEY=sk-ant-your-key`. To keep the key across new windows on Windows, run `setx ANTHROPIC_API_KEY "sk-ant-your-key"` once, then open a new window. (In PowerShell, `set NAME=value` does **not** set an environment variable.)

The same applies to the CLI: `swarmsight run examples/harborview_congestion.md -q "..." -r 10`.

With no flag, SwarmSight uses Claude when it finds credentials and falls back to the offline mock otherwise. The header of the web UI shows which one is active.

| Setting | Default | |
|---|---|---|
| `SWARMSIGHT_MODEL` | `claude-opus-5-5` | Any Claude model ID |
| `SWARMSIGHT_LLM` | `auto` | `auto`, `claude` or `mock` |
| `SWARMSIGHT_DATA` | `data/projects` | Where projects are saved (JSON) |

**Cost:** a run makes about one call per agent per round, plus one call per seed chunk, one per persona and one for the report. Per-agent calls use low effort to keep large swarms affordable. The shared system prompts are cached. Start with ~10 agents and ~5 rounds.

## Web UI

- **Stance:** mean stance with support/oppose shares per round, with injected events marked.
- **Graph:** the extracted knowledge graph. Hover for each entity's summary.
- **Agents:** every persona, its influence, and how its stance moved. Click **Chat** to interview one.
- **Feed:** the full simulated timeline, including world events.
- **Report:** the prediction report, downloadable as Markdown.
- **Chat:** talk to the analyst or any agent.

Link straight to a project and tab with `/#<project-id>/<tab>`.

## API

| Method | Path | |
|---|---|---|
| `POST` | `/api/projects` | `{question, seed, title?, max_agents?}`: creates a project and builds the world in the background |
| `GET` | `/api/projects` | List projects |
| `GET` | `/api/projects/{id}` | Full state: graph, agents, posts, stats, report, `busy` flag |
| `POST` | `/api/projects/{id}/inject` | `{text, round?}`: schedules a world event (defaults to the next round) |
| `POST` | `/api/projects/{id}/run` | `{rounds}`: runs more rounds, then rewrites the report |
| `POST` | `/api/projects/{id}/chat` | `{agent_id?, messages}`: talks to an agent, or to the analyst without `agent_id` |
| `GET` | `/api/projects/{id}/report.md` | The report as Markdown |

## Project layout

```
swarmsight/
  schemas.py     pydantic models, also used as LLM structured-output schemas
  llm.py         ClaudeLLM (Anthropic SDK) and the offline MockLLM
  graph.py       chunking, extraction, merging, the knowledge graph
  personas.py    graph entities -> agent personas
  simulation.py  the world, feeds, memory, rounds, injected events
  report.py      evidence digest, report writing, chat
  project.py     persistence and background jobs
  server.py      FastAPI app
  cli.py         `swarmsight run` / `swarmsight serve`
  web/index.html single-file web UI
tests/           pytest suite (runs offline)
examples/        a fictional sample scenario
```

## Tests

```bash
pytest
```

The suite runs fully offline. It covers graph merging, persona generation, simulation behaviour (reproducibility, injected events, bad model outputs), the report, the HTTP API, and the request that the real Anthropic SDK sends, captured with an in-process transport.

## Limits

A simulation is a structured way to think through how a situation might unfold, not a guarantee. Results depend on the seed material, the model and the number of agents and rounds. The `--mock` mode uses simple heuristics instead of a language model and is meant for demos and testing.

## License

MIT
