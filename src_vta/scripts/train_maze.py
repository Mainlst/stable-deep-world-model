"""
3D Maze 専用の学習起動スクリプト（中身は共通トレーナーを呼び出し）。
"""

from src_vta.scripts.train import main


def run_maze():
    main(default_env="3d_maze")


if __name__ == "__main__":
    run_maze()
