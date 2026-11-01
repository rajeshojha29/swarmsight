import pytest
from fastapi.testclient import TestClient

from conftest import QUESTION
from swarmsight.server import create_app


@pytest.fixture
def client(tmp_path, llm):
    app = create_app(llm, tmp_path)
    with TestClient(app) as c:
        c.store = app.state.store
        yield c


def create(client, seed):
    r = client.post("/api/projects", json={"question": QUESTION, "seed": seed, "max_agents": 8})
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    client.store.wait(pid, 30)
    return pid


def test_health_and_ui(client):
    assert client.get("/api/health").json() == {"ok": True, "llm": "mock"}
    assert "SwarmSight" in client.get("/").text


def test_full_flow(client, seed):
    pid = create(client, seed)
    p = client.get(f"/api/projects/{pid}").json()
    assert p["status"] == "ready", p["message"]
    assert 2 <= len(p["agents"]) <= 8 and p["graph"]["nodes"]

    r = client.post(f"/api/projects/{pid}/inject", json={"text": "The vote is delayed amid protests", "round": 2})
    assert r.status_code == 201

    assert client.post(f"/api/projects/{pid}/run", json={"rounds": 3}).status_code == 202
    client.store.wait(pid, 30)
    p = client.get(f"/api/projects/{pid}").json()
    assert p["status"] == "done", p["message"]
    assert p["rounds"] == 3 and len(p["stats"]) == 4
    assert any(x["kind"] == "event" for x in p["posts"])
    assert p["report"]["markdown"].startswith("# ")
    assert client.get(f"/api/projects/{pid}/report.md").text == p["report"]["markdown"]

    # Past rounds can't take new events.
    assert client.post(f"/api/projects/{pid}/inject", json={"text": "late", "round": 1}).status_code == 400

    agent = p["agents"][0]
    r = client.post(f"/api/projects/{pid}/chat", json={"agent_id": agent["id"],
                                                     "messages": [{"role": "user", "content": "How do you feel?"}]})
    assert r.status_code == 200 and agent["name"] in r.json()["reply"]
    r = client.post(f"/api/projects/{pid}/chat", json={"messages": [{"role": "user", "content": "Why?"}]})
    assert r.status_code == 200 and r.json()["reply"]

    assert [x["id"] for x in client.get("/api/projects").json()] == [pid]


def test_persists_across_restart(tmp_path, llm, seed):
    with TestClient(create_app(llm, tmp_path)) as c:
        c.store = c.app.state.store
        pid = create(c, seed)
    with TestClient(create_app(llm, tmp_path)) as c2:
        assert c2.get(f"/api/projects/{pid}").json()["status"] == "ready"


def test_errors(client, seed):
    assert client.post("/api/projects", json={"question": "Q?", "seed": "   "}).status_code == 400
    assert client.get("/api/projects/nope").status_code == 404
    assert client.get("/api/projects/..%2Fetc").status_code == 404
    pid = create(client, seed)
    assert client.post(f"/api/projects/{pid}/run", json={"rounds": 0}).status_code == 422
    r = client.post(f"/api/projects/{pid}/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 409  # no report yet
    r = client.post(f"/api/projects/{pid}/chat", json={"agent_id": "ghost", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404


def test_seed_without_actors_reports_error(client):
    r = client.post("/api/projects", json={"question": "Q?", "seed": "it rained all day and nothing happened."})
    pid = r.json()["id"]
    client.store.wait(pid, 30)
    p = client.get(f"/api/projects/{pid}").json()
    assert p["status"] == "error" and "no people" in p["message"]
