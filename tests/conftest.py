from pathlib import Path

import pytest

from swarmsight.llm import MockLLM

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "harborview_congestion.md"
QUESTION = "Will the Harborview City Council approve the congestion charge?"


@pytest.fixture
def llm():
    return MockLLM()


@pytest.fixture
def seed():
    return EXAMPLE.read_text(encoding="utf-8")
