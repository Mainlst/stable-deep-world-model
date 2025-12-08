"""
Thin wrapper to run the unified trainer for the Bouncing Balls environment.
"""

from src_vta.scripts.train import main


def entrypoint():
    main(default_env="bouncing_balls")


if __name__ == "__main__":
    entrypoint()
