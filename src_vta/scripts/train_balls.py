"""
Bouncing Balls 専用の学習起動スクリプト（中身は共通トレーナーを呼び出し）。
"""

from src_vta.scripts.train import main


def run_bouncing_balls():
    main(default_env="bouncing_balls")


if __name__ == "__main__":
    run_bouncing_balls()
