"""
Utilities for locating dataset directories.
"""

from pathlib import Path
from typing import Optional, Union

# Default layout under repository root
DEFAULT_MAZE_DIR = Path("data/3d_maze_default")
# Legacy layout kept for backward compatibility
LEGACY_MAZE_DIR = Path("3d_maze_default")


def resolve_maze_data_dir(preferred: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve the base directory for 3D Maze npz files.

    Order of preference:
        1. Explicitly provided path
        2. Default new layout (data/3d_maze_default) if it exists
        3. Legacy layout (./3d_maze_default) if it exists
        4. Default new layout (even if missing) so callers can create it
    """
    if preferred is not None:
        return Path(preferred)

    if DEFAULT_MAZE_DIR.exists():
        return DEFAULT_MAZE_DIR
    if LEGACY_MAZE_DIR.exists():
        return LEGACY_MAZE_DIR
    return DEFAULT_MAZE_DIR
