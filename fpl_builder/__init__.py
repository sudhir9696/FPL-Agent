"""fpl_builder -- build an FPL squad from the public API plus Reddit sentiment."""

from .analysis import ModelConfig, ProjectionModel, best_value, differentials, top_by_position
from .api import FPLClient, FPLError, GameData
from .models import Player, Projection, Squad, Team
from .optimizer import OptimizerError, optimize_squad, pick_best_xi, suggest_transfers
from .reddit import RedditClient, RedditError, cross_reference, extract_player_buzz, gather_threads

__version__ = "0.1.0"

__all__ = [
    "FPLClient", "FPLError", "GameData",
    "ModelConfig", "ProjectionModel",
    "Player", "Projection", "Squad", "Team",
    "optimize_squad", "pick_best_xi", "suggest_transfers", "OptimizerError",
    "RedditClient", "RedditError", "gather_threads", "extract_player_buzz", "cross_reference",
    "best_value", "differentials", "top_by_position",
    "__version__",
]
