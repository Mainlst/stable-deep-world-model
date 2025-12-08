from .base import EnvTrainContext, EnvironmentAdapter, TrainBatchProvider
from .balls import BouncingBallsAdapter
from .maze import MazeAdapter

ADAPTERS = {
    BouncingBallsAdapter.name: BouncingBallsAdapter(),
    MazeAdapter.name: MazeAdapter(),
}

__all__ = [
    "EnvTrainContext",
    "EnvironmentAdapter",
    "TrainBatchProvider",
    "ADAPTERS",
    "BouncingBallsAdapter",
    "MazeAdapter",
]
