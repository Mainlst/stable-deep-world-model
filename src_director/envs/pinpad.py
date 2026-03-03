import collections
import gym
import numpy as np


# -----------------------------
# Core Gym Env (old Gym API)
# -----------------------------
class PinPadEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    COLORS = {
        "1": (255, 0, 0),
        "2": (0, 255, 0),
        "3": (0, 0, 255),
        "4": (255, 255, 0),
        "5": (255, 0, 255),
        "6": (0, 255, 255),
        "7": (128, 0, 128),
        "8": (0, 128, 128),
    }

    def __init__(self, task: str, length: int = 10_000, seed: int = 0):
        super().__init__()
        assert length > 0

        layout_map = {
            "three": LAYOUT_THREE,
            "four": LAYOUT_FOUR,
            "five": LAYOUT_FIVE,
            "six": LAYOUT_SIX,
            "seven": LAYOUT_SEVEN,
            "eight": LAYOUT_EIGHT,
        }
        if task not in layout_map:
            raise ValueError(f"Unknown task={task}. Choose from {list(layout_map.keys())}.")

        layout = layout_map[task]
        self.layout = np.array([list(line) for line in layout.split("\n")]).T
        if self.layout.shape != (16, 14):
            raise ValueError(f"layout shape must be (16, 14), got {self.layout.shape}")

        self.length = int(length)
        self.random = np.random.RandomState(seed)

        self.pads = set(self.layout.flatten().tolist()) - set(["*", " ", "#"])
        self.target = tuple(sorted(self.pads))

        self.spawns = []
        for (x, y), ch in np.ndenumerate(self.layout):
            if ch != "#":
                self.spawns.append((int(x), int(y)))

        self.sequence = collections.deque(maxlen=len(self.target))
        self.player = None
        self.steps = 0
        self.done = False
        self.countdown = 0

        # Gym spaces
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )

    def seed(self, seed=None):
        if seed is None:
            return
        self.random = np.random.RandomState(int(seed))

    def reset(self):
        self.player = self.spawns[self.random.randint(len(self.spawns))]
        self.sequence.clear()
        self.steps = 0
        self.done = False
        self.countdown = 0
        return self.render(mode="rgb_array")

    def step(self, action: int):
        if self.done:
            # 旧Dreamer系の実装に合わせて: done後にstepされたらリセット相当を返すより、
            # 利用側が reset() する想定で、そのまま done を維持します。
            obs = self.render(mode="rgb_array")
            return obs, 0.0, True, {"is_terminal": False}

        # countdown handling
        if self.countdown:
            self.countdown -= 1
            if self.countdown == 0:
                self.player = self.spawns[self.random.randint(len(self.spawns))]
                self.sequence.clear()

        reward = 0.0

        # 0: stay, 1: right, 2: left, 3: down, 4: up
        move = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)][int(action)]
        px, py = self.player
        x = int(np.clip(px + move[0], 0, 15))
        y = int(np.clip(py + move[1], 0, 13))

        tile = self.layout[x, y]
        if tile != "#":
            self.player = (x, y)

        if tile in self.pads:
            if not self.sequence or self.sequence[-1] != tile:
                self.sequence.append(tile)

        if tuple(self.sequence) == self.target and not self.countdown:
            reward += 10.0
            self.countdown = 10

        self.steps += 1
        self.done = (self.steps >= self.length)

        obs = self.render(mode="rgb_array")
        info = {"is_terminal": False}
        return obs, float(reward), bool(self.done), info

    def render(self, mode="rgb_array"):
        # produce 64x64x3 uint8
        grid = np.zeros((16, 16, 3), np.uint8) + 255
        white = np.array([255, 255, 255], dtype=np.float32)

        if self.countdown:
            grid[:] = (223, 255, 223)

        current = self.layout[self.player[0], self.player[1]] if self.player is not None else " "

        for (x, y), ch in np.ndenumerate(self.layout):
            if ch == "#":
                grid[x, y] = (192, 192, 192)
            elif ch in self.pads:
                color = np.array(self.COLORS[ch], dtype=np.float32)
                color = color if ch == current else (10 * color + 90 * white) / 100.0
                grid[x, y] = color.astype(np.uint8)

        if self.player is not None:
            grid[self.player] = (0, 0, 0)

        grid[:, -2:] = (192, 192, 192)
        for i, ch in enumerate(self.sequence):
            r = 2 * i + 1
            if r < 16:
                grid[r, -2] = self.COLORS[ch]

        image = np.repeat(np.repeat(grid, 4, axis=0), 4, axis=1)
        return image.transpose((1, 0, 2))


# -----------------------------------
# DreamerV2-style wrapper (your ref)
# -----------------------------------
class PinPad:
    def __init__(
        self,
        task,
        obs_key="image",
        act_key="action",
        size=(64, 64),
        seed=0,
        length=10_000,
    ):
        self._env = PinPadEnv(task=task, length=length, seed=seed)
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key
        self._size = tuple(size)
        self._gray = False

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def observation_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}
        return gym.spaces.Dict(
            {
                **spaces,
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            }
        )

    @property
    def action_space(self):
        space = self._env.action_space
        # DreamerV2が見る属性を付ける慣例
        space.discrete = True
        return space

    def _resize_if_needed(self, img: np.ndarray) -> np.ndarray:
        if img.shape[0] == self._size[0] and img.shape[1] == self._size[1]:
            return img
        # Pillow があればリサイズ（なければエラー）
        try:
            from PIL import Image
        except Exception as e:
            raise RuntimeError(
                "Resizing requires Pillow. Install with: pip install pillow"
            ) from e
        pil = Image.fromarray(img)
        pil = pil.resize((self._size[1], self._size[0]), resample=Image.NEAREST)
        return np.asarray(pil, dtype=np.uint8)

    def step(self, action):
        # 参照クラス同様、actionはそのまま渡す（Discrete想定）
        obs, reward, done, info = self._env.step(action)

        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        else:
            obs = dict(obs)

        # size指定があれば image をリサイズ
        if self._obs_key in obs and isinstance(obs[self._obs_key], np.ndarray):
            obs[self._obs_key] = self._resize_if_needed(obs[self._obs_key])

        obs["is_first"] = False
        obs["is_last"] = bool(done)
        obs["is_terminal"] = bool(info.get("is_terminal", False))
        return obs, float(reward), bool(done), info

    def reset(self):
        obs = self._env.reset()

        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        else:
            obs = dict(obs)

        if self._obs_key in obs and isinstance(obs[self._obs_key], np.ndarray):
            obs[self._obs_key] = self._resize_if_needed(obs[self._obs_key])

        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        return obs


# --- layouts (unchanged) ---
LAYOUT_THREE = """
################
#1111      3333#
#1111      3333#
#1111      3333#
#1111      3333#
#              #
#              #
#              #
#              #
#     2222     #
#     2222     #
#     2222     #
#     2222     #
################
""".strip("\n")

LAYOUT_FOUR = """
################
#1111      4444#
#1111      4444#
#1111      4444#
#1111      4444#
#              #
#              #
#              #
#              #
#3333      2222#
#3333      2222#
#3333      2222#
#3333      2222#
################
""".strip("\n")

LAYOUT_FIVE = """
################
#          4444#
#111       4444#
#111       4444#
#111           #
#111        555#
#           555#
#           555#
#333        555#
#333           #
#333       2222#
#333       2222#
#          2222#
################
""".strip("\n")

LAYOUT_SIX = """
################
#111        555#
#111        555#
#111        555#
#              #
#33          66#
#33          66#
#33          66#
#33          66#
#              #
#444        222#
#444        222#
#444        222#
################
""".strip("\n")

LAYOUT_SEVEN = """
################
#111        444#
#111        444#
#11          44#
#              #
#33          55#
#33          55#
#33          55#
#33          55#
#              #
#66          22#
#666  7777  222#
#666  7777  222#
################
""".strip("\n")

LAYOUT_EIGHT = """
################
#111  8888  444#
#111  8888  444#
#11          44#
#              #
#33          55#
#33          55#
#33          55#
#33          55#
#              #
#66          22#
#666  7777  222#
#666  7777  222#
################
""".strip("\n")