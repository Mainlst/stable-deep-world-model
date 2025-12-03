'''
3D_Maze環境の実装モジュール
(実装が上手くできていないため，現在使用予定なし)
'''

import numpy as np
import torch
from tqdm import tqdm

class Raycaster:
    """
    NumPyを用いた簡易的なRaycastingレンダリングエンジン
    """
    def __init__(self, map_data, resolution=32, fov=60):
        self.map_data = map_data
        self.height, self.width = map_data.shape
        self.resolution = resolution
        self.fov = np.deg2rad(fov)
        self.half_fov = self.fov / 2
        # 壁IDに対応する色 (R, G, B) 0-1
        self.colors = {
            0: [0.0, 0.0, 0.0],
            1: [0.8, 0.1, 0.1], 2: [0.1, 0.8, 0.1], 3: [0.1, 0.1, 0.8],
            4: [0.8, 0.8, 0.1], 5: [0.1, 0.8, 0.8], 6: [0.8, 0.1, 0.8],
            9: [0.5, 0.5, 0.5],
        }

    def render(self, pos_x, pos_y, angle):
        # レイの角度計算
        ray_angles = (angle - self.half_fov) + (np.arange(self.resolution) / self.resolution) * self.fov
        
        map_x = int(pos_x)
        map_y = int(pos_y)
        
        ray_dir_x = np.cos(ray_angles)
        ray_dir_y = np.sin(ray_angles)
        
        ray_dir_x[np.abs(ray_dir_x) < 1e-6] = 1e-6
        ray_dir_y[np.abs(ray_dir_y) < 1e-6] = 1e-6

        delta_dist_x = np.abs(1.0 / ray_dir_x)
        delta_dist_y = np.abs(1.0 / ray_dir_y)
        
        step_x = np.sign(ray_dir_x).astype(int)
        step_y = np.sign(ray_dir_y).astype(int)
        
        side_dist_x = (np.where(ray_dir_x < 0, pos_x - np.floor(pos_x), np.ceil(pos_x) - pos_x)) * delta_dist_x
        side_dist_y = (np.where(ray_dir_y < 0, pos_y - np.floor(pos_y), np.ceil(pos_y) - pos_y)) * delta_dist_y
        
        hit = np.zeros(self.resolution, dtype=bool)
        side = np.zeros(self.resolution, dtype=int)
        cur_map_x = np.full(self.resolution, map_x, dtype=int)
        cur_map_y = np.full(self.resolution, map_y, dtype=int)
        wall_types = np.zeros(self.resolution, dtype=int)
        
        active_rays = np.ones(self.resolution, dtype=bool)
        
        # Raycasting Loop
        while np.any(active_rays):
            mask_x = (side_dist_x < side_dist_y) & active_rays
            mask_y = (~mask_x) & active_rays
            
            side_dist_x[mask_x] += delta_dist_x[mask_x]
            cur_map_x[mask_x] += step_x[mask_x]
            side[mask_x] = 0
            
            side_dist_y[mask_y] += delta_dist_y[mask_y]
            cur_map_y[mask_y] += step_y[mask_y]
            side[mask_y] = 1
            
            out_of_bounds = (cur_map_x < 0) | (cur_map_x >= self.width) | (cur_map_y < 0) | (cur_map_y >= self.height)
            active_rays[out_of_bounds] = False
            
            in_bounds = ~out_of_bounds
            hit_indices = in_bounds & active_rays
            
            safe_x = np.clip(cur_map_x, 0, self.width-1)
            safe_y = np.clip(cur_map_y, 0, self.height-1)
            current_walls = self.map_data[safe_y, safe_x]
            
            hit_wall = (current_walls > 0) & hit_indices
            wall_types[hit_wall] = current_walls[hit_wall]
            active_rays[hit_wall] = False 
            
        perp_wall_dist = np.zeros(self.resolution)
        mask_side0 = (side == 0)
        perp_wall_dist[mask_side0] = (cur_map_x[mask_side0] - pos_x + (1 - step_x[mask_side0]) / 2) / ray_dir_x[mask_side0]
        mask_side1 = (side == 1)
        perp_wall_dist[mask_side1] = (cur_map_y[mask_side1] - pos_y + (1 - step_y[mask_side1]) / 2) / ray_dir_y[mask_side1]
        
        image = np.zeros((self.resolution, self.resolution, 3))
        line_heights = (self.resolution / (perp_wall_dist + 1e-6)).astype(int)
        
        for x in range(self.resolution):
            h = line_heights[x]
            draw_start = max(0, -h // 2 + self.resolution // 2)
            draw_end = min(self.resolution, h // 2 + self.resolution // 2)
            
            wall_color = self.colors.get(wall_types[x], [1.0, 1.0, 1.0])
            brightness = 1.0 / (1.0 + perp_wall_dist[x] * 0.1)
            side_shadow = 0.7 if side[x] == 1 else 1.0
            
            color = np.array(wall_color) * brightness * side_shadow
            image[draw_start:draw_end, x] = color
            
            if draw_start > 0: image[:draw_start, x] = [0.1, 0.1, 0.1]
            if draw_end < self.resolution: image[draw_end:, x] = [0.2, 0.2, 0.2]

        return image

class MazeEnv:
    def __init__(self, resolution=32):
        self.map_str = [
            "1111111111111111",
            "1000000012000002",
            "1011111012022202",
            "1013331000020002",
            "1013003066020444",
            "1000000060000404",
            "1111155560660404",
            "1000050000600004",
            "1022050555664444",
            "1020000000000004",
            "1111111111111114"
        ]
        self.map_data = np.array([[int(c) for c in row] for row in self.map_str])
        self.height, self.width = self.map_data.shape
        self.resolution = resolution
        self.renderer = Raycaster(self.map_data, resolution=resolution)
        
        self.pos_x = 1.5
        self.pos_y = 1.5
        self.angle = 0.0
        
        self.move_speed = 0.4
        self.rot_speed = 0.3 # 約17度

    def reset(self):
        while True:
            x = np.random.randint(1, self.width - 1)
            y = np.random.randint(1, self.height - 1)
            if self.map_data[y, x] == 0:
                self.pos_x = x + 0.5
                self.pos_y = y + 0.5
                # 初期向きは 0, 90, 180, 270 度のいずれかにスナップさせると安定しやすい
                self.angle = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2])
                # 少しノイズを入れる
                self.angle += np.random.normal(0, 0.05)
                break
        return self.get_observation()

    def _is_wall(self, x, y):
        # 座標が壁の中かどうか判定
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return self.map_data[int(y), int(x)] > 0

    def get_permissible_actions(self):
        """
        論文の制約に基づき、現在の状態で許可されるアクションのリストを返す
        0: Forward, 1: Left, 2: Right
        """
        # 現在の進行方向ベクトル
        dx = np.cos(self.angle) * self.move_speed
        dy = np.sin(self.angle) * self.move_speed
        
        # 1. 前方に壁があるかチェック
        # 少し先を見て壁なら前進禁止
        check_dist = 0.6 
        front_x = self.pos_x + np.cos(self.angle) * check_dist
        front_y = self.pos_y + np.sin(self.angle) * check_dist
        is_front_blocked = self._is_wall(front_x, front_y)
        
        # 2. 左右に道があるか（交差点判定）チェック
        # 現在の向きから +/- 90度 方向の壁チェック
        left_angle = self.angle - np.pi/2
        right_angle = self.angle + np.pi/2
        
        left_x = self.pos_x + np.cos(left_angle) * check_dist
        left_y = self.pos_y + np.sin(left_angle) * check_dist
        is_left_open = not self._is_wall(left_x, left_y)
        
        right_x = self.pos_x + np.cos(right_angle) * check_dist
        right_y = self.pos_y + np.sin(right_angle) * check_dist
        is_right_open = not self._is_wall(right_x, right_y)
        
        actions = []
        
        if is_front_blocked:
            # 前が壁なら、必ず回転しなければならない（廊下でも突き当たりなら回転可）
            actions.append(1) # Left
            actions.append(2) # Right
        else:
            # 前が空いている場合
            actions.append(0) # Forward
            
            # 交差点（左右どちらかが開いている）なら回転も許可
            if is_left_open or is_right_open:
                actions.append(1)
                actions.append(2)
            else:
                # 左右が壁で、前が空いている ＝ 典型的な「廊下」
                # この場合、回転アクションを追加しない ＝ 直進のみ (Constraint: not allowed to turn around)
                pass
                
        return actions

    def step(self, action):
        next_x = self.pos_x
        next_y = self.pos_y
        next_angle = self.angle

        if action == 0:   # Forward
            next_x += np.cos(self.angle) * self.move_speed
            next_y += np.sin(self.angle) * self.move_speed
        elif action == 1: # Left
            next_angle -= self.rot_speed
        elif action == 2: # Right
            next_angle += self.rot_speed

        # 衝突判定と更新
        if not self._is_wall(next_x, next_y):
            self.pos_x = next_x
            self.pos_y = next_y
        
        self.angle = next_angle
        
        # ジッターノイズ (論文再現)
        self.pos_x += np.random.normal(0, 0.005)
        self.pos_y += np.random.normal(0, 0.005)
        self.angle += np.random.normal(0, 0.01)

        return self.get_observation()

    def get_observation(self):
        return self.renderer.render(self.pos_x, self.pos_y, self.angle)

    def is_intersection(self):
        # 簡易ヘルパー（今回はget_permissible_actionsで精密にやるので使わないが互換性のため残す）
        gx, gy = int(self.pos_x), int(self.pos_y)
        openings = 0
        if not self._is_wall(gx, gy):
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                 if not self._is_wall(gx+dx, gy+dy):
                     openings += 1
        return openings >= 3