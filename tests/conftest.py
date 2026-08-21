import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_agent.analysis import ModelConfig, ProjectionModel
from fpl_agent.api import GameData

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def data():
    return GameData.from_files(FIXTURES / "bootstrap.json", FIXTURES / "fixtures.json")


@pytest.fixture(scope="session")
def model(data):
    return ProjectionModel(data, ModelConfig(horizon=5))


@pytest.fixture(scope="session")
def projections(data, model):
    return model.project_all(start_gw=data.next_gameweek)
