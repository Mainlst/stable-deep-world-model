"""
Thin wrapper to run the unified trainer for the 3D Maze environment.
"""

from src_vta.scripts.train import main


def entrypoint():
    main(default_env="3d_maze")


if __name__ == "__main__":
    entrypoint()
