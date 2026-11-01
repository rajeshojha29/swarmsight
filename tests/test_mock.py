"""The offline mock is heuristic, but it should not turn metrics or rules into agents."""

from swarmsight.graph import build_graph
from swarmsight.llm import _classify, _lexicon_score
from swarmsight.personas import build_personas
from swarmsight.schemas import Injection
from swarmsight.simulation import World, run_simulation

SEED = """Malwa Chemicals plans to convert its spent liquor into poly aluminium chloride for water treatment.
The project needs Rs 17.6 crore and shows a projected IRR of 24 percent. Projected EBITDA margins depend on
the price buyers pay. NALCO supplies alumina hydrate and has offered a long-term contract. GACL is a rival
producer and warns that prices could fall. SBI is willing to fund the plant if MPPCB approves the permit.
Under the Hazardous Waste Rules, MPPCB must clear the process before trial runs begin. Textile Buyers
Association members say they would sign letters of intent at Rs 26 per kg if the product meets the ISI
standard. Anil Mehta, who runs Malwa Chemicals, backs the plan. Consultant Priya Nair fears a delay."""

QUESTION = "Should Malwa Chemicals invest in the PAC plant?"


def test_metrics_and_rules_are_not_actors():
    for name in ("IRR", "EBITDA", "ISI", "Projected EBITDA", "Hazardous Waste Rules"):
        assert _classify(name) == "concept", name
    for name in ("NALCO", "GACL", "SBI", "MPPCB"):
        assert _classify(name) == "organization", name


def test_inflected_words_count():
    assert _lexicon_score("The lab test passes and MPPCB approves the permit.") > 0
    assert _lexicon_score("The buyer meets the standard and confirmed the order.") > 0
    assert _lexicon_score("The permit was denied after the test failed.") < 0


def test_pac_seed_agents(llm):
    graph = build_graph(SEED, llm)
    names = {p.name for p in build_personas(graph, QUESTION, llm, max_agents=12)}
    assert {"NALCO", "GACL", "SBI", "MPPCB", "Malwa Chemicals", "Priya Nair"} <= names
    assert not names & {"IRR", "EBITDA", "ISI", "Projected EBITDA", "Hazardous Waste Rules"}


def test_good_news_event_raises_stance(llm):
    graph = build_graph(SEED, llm)
    personas = build_personas(graph, QUESTION, llm, max_agents=12)
    base = run_simulation(World.create(QUESTION, personas), llm, 4)
    shocked = World.create(QUESTION, personas)
    shocked.injections.append(Injection(round=3, text="The lab test passes IS 15573 and MPPCB approves the permit"))
    run_simulation(shocked, llm, 4)
    assert shocked.stats[3].mean_stance > base.stats[3].mean_stance + 0.1


def test_mock_report_is_labelled(llm):
    from swarmsight.report import write_report
    graph = build_graph(SEED, llm)
    world = run_simulation(World.create(QUESTION, build_personas(graph, QUESTION, llm, max_agents=8)), llm, 2)
    report = write_report(world, llm, graph)
    assert report.headline.startswith("[Mock demo, not AI]")
    assert "not a language model" in report.markdown
