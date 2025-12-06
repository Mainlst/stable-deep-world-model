import os

# NOTE: Env MUJOCO_GL=egl is necessary for headless hardware rendering on GPU,
# but breaks when running on a CPU machine. Alternatively set MUJOCO_GL=osmesa.
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'egl'

from . import tasks

def register():
    try:
        # Register gym environments, if gym is available

        from typing import Callable
        from functools import partial as f

        import dm_env
        import gymnasium as gym
        from gymnasium.envs.registration import register as gym_register

        from .gym_wrappers import GymWrapper

        def _make_gym_env(dm_task: Callable[[], dm_env.Environment], **kwargs):
            dmenv = dm_task(**kwargs)
            return GymWrapper(dmenv)

        sizes = {
            '9x9': tasks.memory_maze_9x9,
            '11x11': tasks.memory_maze_11x11,
            '13x13': tasks.memory_maze_13x13,
            '15x15': tasks.memory_maze_15x15,
        }

        for key, dm_task in sizes.items():
            # Image-only obs space
            gym_register(id=f'MemoryMaze-{key}-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True))  # Standard
            gym_register(id=f'MemoryMaze-{key}-Vis-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True, good_visibility=True))  # Easily visible targets
            gym_register(id=f'MemoryMaze-{key}-HD-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True, camera_resolution=256))  # High-res camera
            gym_register(id=f'MemoryMaze-{key}-Top-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True, camera_resolution=256, top_camera=True))  # Top-down camera
            
            # Extra global observables (dict obs space)
            gym_register(id=f'MemoryMaze-{key}-ExtraObs-v0', entry_point=f(_make_gym_env, dm_task, global_observables=True))
            gym_register(id=f'MemoryMaze-{key}-ExtraObs-Vis-v0', entry_point=f(_make_gym_env, dm_task, global_observables=True, good_visibility=True))
            gym_register(id=f'MemoryMaze-{key}-ExtraObs-Top-v0', entry_point=f(_make_gym_env, dm_task, global_observables=True, camera_resolution=256, top_camera=True))
            
            # Oracle observables with shortest path shown
            gym_register(id=f'MemoryMaze-{key}-Oracle-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True, global_observables=True, show_path=True))
            gym_register(id=f'MemoryMaze-{key}-Oracle-Top-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True, global_observables=True, show_path=True, camera_resolution=256, top_camera=True))
            gym_register(id=f'MemoryMaze-{key}-Oracle-ExtraObs-v0', entry_point=f(_make_gym_env, dm_task, global_observables=True, show_path=True))
            
            # High control frequency
            gym_register(id=f'MemoryMaze-{key}-HiFreq-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True, control_freq=40))
            gym_register(id=f'MemoryMaze-{key}-HiFreq-Vis-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True, control_freq=40, good_visibility=True))
            gym_register(id=f'MemoryMaze-{key}-HiFreq-HD-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True, control_freq=40, camera_resolution=256))

            # Six colors even for smaller mazes
            gym_register(id=f'MemoryMaze-{key}-6CL-v0', entry_point=f(_make_gym_env, dm_task, randomize_colors=True, image_only_obs=True))
            gym_register(id=f'MemoryMaze-{key}-6CL-Top-v0', entry_point=f(_make_gym_env, dm_task, randomize_colors=True, image_only_obs=True, camera_resolution=256, top_camera=True))
            gym_register(id=f'MemoryMaze-{key}-6CL-ExtraObs-v0', entry_point=f(_make_gym_env, dm_task, randomize_colors=True, global_observables=True))
            
    except ImportError:
        print('memory_maze: gym environments not registered.')
        raise
