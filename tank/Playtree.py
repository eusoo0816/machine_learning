import pygame
import random
import math
import sys
import os
import joblib
import pandas as pd

# ===================== 基本設定 =====================
SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 720

WORLD_WIDTH  = SCREEN_WIDTH
WORLD_HEIGHT = SCREEN_HEIGHT

FPS = 60
GAME_TIME_SECONDS = 180  # 180 秒對戰

TANK_SPEED = 8
BULLET_SPEED = 14

MAX_LIFE = 3
MAX_FUEL = 100
MAX_AMMO = 20

FUEL_SUPPLY_AMOUNT = 30
AMMO_SUPPLY_AMOUNT = 20

SUPPLY_RESPAWN_FRAMES = 30

# 顏色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 128, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREY = (120, 120, 120)
DARKGREY = (50, 50, 50)

# ===================== 模型路徑 =====================
MODEL_DIR = r"C:\Users\kai\Desktop\Machinelearning\TANK\tree\treedata"
GOAL_MODEL_PATH = os.path.join(MODEL_DIR, "goal_model.joblib")
MOVE_FIGHT_MODEL_PATH = os.path.join(MODEL_DIR, "move_fight_model.joblib")
MOVE_SUPPLY_MODEL_PATH = os.path.join(MODEL_DIR, "move_supply_model.joblib")
FIRE_MODEL_PATH = os.path.join(MODEL_DIR, "fire_model.joblib")

# ===================== 小工具函式 =====================
def clamp(value, mn, mx):
    return max(mn, min(mx, value))

def angle_to_vector(angle_deg):
    rad = math.radians(angle_deg)
    return math.cos(rad), math.sin(rad)

def vector_to_angle_deg(dx, dy):
    ang = math.degrees(math.atan2(dy, dx))
    if ang < 0:
        ang += 360
    return ang

def angle_diff_deg(a, b):
    d = (a - b) % 360
    if d > 180:
        d = 360 - d
    return abs(d)

def dist_xy(x1, y1, x2, y2):
    return math.hypot(x2-x1, y2-y1)

def safe_unit(ax, ay, eps=1e-9):
    n = math.hypot(ax, ay)
    if n < eps:
        return 0.0, 0.0
    return ax / n, ay / n

def has_line_of_sight(x1, y1, x2, y2, walls, step=12):
    dx = x2 - x1
    dy = y2 - y1
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return True
    ux, uy = dx/d, dy/d
    steps = int(d // step)
    for i in range(1, steps):
        px = x1 + ux * (i*step)
        py = y1 + uy * (i*step)
        for w in walls:
            if (not w.is_destroyed()) and w.rect.collidepoint(px, py):
                return False
    return True

def first_wall_hit_point(x1, y1, x2, y2, walls, step=10):
    dx = x2 - x1
    dy = y2 - y1
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return False, -1, -1, -1

    ux, uy = dx / d, dy / d
    steps = int(d // step)

    for i in range(1, steps + 1):
        px = x1 + ux * (i * step)
        py = y1 + uy * (i * step)
        for w in walls:
            if not w.is_destroyed() and w.rect.collidepoint(px, py):
                return True, px, py, math.hypot(px - x1, py - y1)

    return False, -1, -1, -1

# ===================== 隨機復活點（全域函式） =====================
def random_respawn_pos(walls, supplies, other_tanks, tank_radius, margin=10, min_dist_other=140, max_tries=400):
    """
    找一個安全隨機點：
    - 不跟牆重疊（含邊界牆）
    - 不壓到補給
    - 距離其他活著的坦克要夠遠
    """
    for _ in range(max_tries):
        x = random.randint(tank_radius + margin, WORLD_WIDTH  - tank_radius - margin)
        y = random.randint(tank_radius + margin, WORLD_HEIGHT - tank_radius - margin)

        t_rect = pygame.Rect(0, 0, tank_radius * 2, tank_radius * 2)
        t_rect.center = (x, y)

        ok = True

        # 不能撞牆
        for w in walls:
            if w.is_destroyed():
                continue
            if t_rect.colliderect(w.rect):
                ok = False
                break
        if not ok:
            continue

        # 不能壓到補給
        for s in supplies:
            if s.active and t_rect.colliderect(s.rect):
                ok = False
                break
        if not ok:
            continue

        # 離其他坦克遠一點
        for tk in other_tanks:
            if not tk.alive:
                continue
            if dist_xy(x, y, tk.x, tk.y) < (min_dist_other + tank_radius):
                ok = False
                break
        if not ok:
            continue

        return x, y

    # fallback
    x = random.randint(tank_radius + margin, WORLD_WIDTH  - tank_radius - margin)
    y = random.randint(tank_radius + margin, WORLD_HEIGHT - tank_radius - margin)
    return x, y


# ===================== 類別定義 =====================
class Tank:
    def __init__(self, x, y, color, team_name):
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y

        self.color = color
        self.team_name = team_name

        self.body_angle = 0
        self.turret_angle = 0
        self.radius = 20

        self.life = MAX_LIFE
        self.fuel = MAX_FUEL
        self.ammo = MAX_AMMO
        self.alive = True

        self.move_forward = False
        self.move_backward = False
        self.move_left = False
        self.move_right = False

    def vx(self):
        return self.x - self.prev_x

    def vy(self):
        return self.y - self.prev_y

    def rect(self):
        r = pygame.Rect(0, 0, self.radius * 2, self.radius * 2)
        r.center = (self.x, self.y)
        return r

    def update(self, walls):
        if not self.alive:
            return

        self.prev_x, self.prev_y = self.x, self.y

        dx = 0
        dy = 0

        if self.fuel > 0:
            if self.move_forward:  dy -= TANK_SPEED
            if self.move_backward: dy += TANK_SPEED
            if self.move_left:     dx -= TANK_SPEED
            if self.move_right:    dx += TANK_SPEED

        if dx != 0 or dy != 0:
            self.fuel = max(0, self.fuel - 0.1)

        old_x, old_y = self.x, self.y

        self.x += dx
        self.y += dy

        self.x = clamp(self.x, self.radius, WORLD_WIDTH  - self.radius)
        self.y = clamp(self.y, self.radius, WORLD_HEIGHT - self.radius)

        # ---- 碰牆回退（避免穿牆）----
        t_rect = self.rect()
        for w in walls:
            if w.is_destroyed():
                continue
            if t_rect.colliderect(w.rect):
                self.x, self.y = old_x, old_y
                break

    def draw(self, surface, camera_offset):
        if not self.alive:
            return

        cx = int(self.x - camera_offset[0])
        cy = int(self.y - camera_offset[1])

        body_surf = pygame.Surface((40, 30), pygame.SRCALPHA)
        pygame.draw.rect(body_surf, self.color, (0, 0, 40, 30))
        rotated_body = pygame.transform.rotate(body_surf, -self.body_angle)
        rect = rotated_body.get_rect(center=(cx, cy))
        surface.blit(rotated_body, rect.topleft)

        tx, ty = angle_to_vector(self.turret_angle)
        gun_len = 30
        end_x = cx + int(tx * gun_len)
        end_y = cy + int(ty * gun_len)
        pygame.draw.line(surface, YELLOW, (cx, cy), (end_x, end_y), 4)

        bar_w = 40
        pygame.draw.rect(surface, DARKGREY, (cx - bar_w // 2, cy - 40, bar_w, 5))
        pygame.draw.rect(surface, RED, (cx - bar_w // 2, cy - 40, int(bar_w * (self.life / MAX_LIFE)), 5))

        pygame.draw.rect(surface, DARKGREY, (cx - bar_w // 2, cy - 34, bar_w, 5))
        pygame.draw.rect(surface, GREEN, (cx - bar_w // 2, cy - 34, int(bar_w * (self.fuel / MAX_FUEL)), 5))

        pygame.draw.rect(surface, DARKGREY, (cx - bar_w // 2, cy - 28, bar_w, 5))
        pygame.draw.rect(surface, BLUE, (cx - bar_w // 2, cy - 28, int(bar_w * (self.ammo / MAX_AMMO)), 5))

    def respawn(self, x, y):
        self.x = x
        self.y = y
        self.prev_x, self.prev_y = x, y
        self.life = MAX_LIFE
        self.fuel = MAX_FUEL
        self.ammo = MAX_AMMO
        self.alive = True
        self.body_angle = 0
        self.turret_angle = 0

    def hit_by_bullet(self):
        if not self.alive:
            return False
        self.life -= 1
        if self.life <= 0:
            self.alive = False
            return True
        return False


class Bullet:
    def __init__(self, x, y, angle_deg, team_name):
        self.x = x
        self.y = y
        self.angle = angle_deg
        self.team_name = team_name
        self.radius = 4
        self.alive = True

    def update(self):
        vx, vy = angle_to_vector(self.angle)
        self.x += vx * BULLET_SPEED
        self.y += vy * BULLET_SPEED
        if self.x < 0 or self.x > WORLD_WIDTH or self.y < 0 or self.y > WORLD_HEIGHT:
            self.alive = False

    def draw(self, surface, camera_offset):
        if not self.alive:
            return
        cx = int(self.x - camera_offset[0])
        cy = int(self.y - camera_offset[1])
        pygame.draw.circle(surface, YELLOW, (cx, cy), self.radius)


class Wall:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.hp = 3

    def is_destroyed(self):
        return self.hp <= 0

    def hit(self):
        if self.hp > 0:
            self.hp -= 1

    def draw(self, surface, camera_offset):
        if self.is_destroyed():
            return
        alpha = 255 if self.hp == 3 else (170 if self.hp == 2 else 85)
        wall_surf = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        wall_surf.fill((*GREY, alpha))
        surface.blit(wall_surf, (self.rect.x - camera_offset[0], self.rect.y - camera_offset[1]))


class Supply:
    def __init__(self, supply_type, x, y):
        self.type = supply_type
        self.rect = pygame.Rect(x, y, 30, 30)
        self.active = True
        self.respawn_timer = 0

    def update(self):
        if not self.active:
            self.respawn_timer -= 1
            if self.respawn_timer <= 0:
                self.rect.x = random.randint(0, WORLD_WIDTH - self.rect.width)
                self.rect.y = random.randint(0, WORLD_HEIGHT - self.rect.height)
                self.active = True

    def consume(self):
        self.active = False
        self.respawn_timer = SUPPLY_RESPAWN_FRAMES

    def draw(self, surface, camera_offset):
        if not self.active:
            return
        x = self.rect.x - camera_offset[0]
        y = self.rect.y - camera_offset[1]
        color = GREEN if self.type == "fuel" else BLUE
        pygame.draw.rect(surface, color, (x, y, self.rect.width, self.rect.height))
        font = pygame.font.SysFont(None, 18)
        text = "F" if self.type == "fuel" else "A"
        t_surf = font.render(text, True, WHITE)
        t_rect = t_surf.get_rect(center=(x + self.rect.width // 2, y + self.rect.height // 2))
        surface.blit(t_surf, t_rect)


# ===================== 模型 AI：讀 joblib + 產生 action =====================
MOVE_STOP, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT = 0, 1, 2, 3, 4

class JoblibPolicy:
    def __init__(self, path: str):
        self.path = path
        self.model = None
        self.feature_cols = None
        self.ok = False
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            print(f"[WARN] model not found: {self.path}")
            return
        obj = joblib.load(self.path)
        self.model = obj.get("model", None)
        self.feature_cols = obj.get("feature_cols", None)
        if self.model is None or self.feature_cols is None:
            print(f"[WARN] invalid model file: {self.path}")
            return
        self.ok = True
        print(f"[INFO] loaded model: {os.path.basename(self.path)} (features={len(self.feature_cols)})")

    def predict_one(self, feat_dict: dict, default_value=0):
        if not self.ok:
            return None
        row = {}
        for c in self.feature_cols:
            row[c] = feat_dict.get(c, default_value)
        X = pd.DataFrame([row], columns=self.feature_cols)
        return int(self.model.predict(X)[0])

class ModelTankAI:
    def __init__(self):
        self.goal_model = JoblibPolicy(GOAL_MODEL_PATH)
        self.move_fight_model = JoblibPolicy(MOVE_FIGHT_MODEL_PATH)
        self.move_supply_model = JoblibPolicy(MOVE_SUPPLY_MODEL_PATH)
        self.fire_model = JoblibPolicy(FIRE_MODEL_PATH)

    def _nearest_threat_bullet(self, tank, bullets):
        min_d = 1e9
        min_ang = -1
        cnt = 0
        for b in bullets:
            if (not b.alive) or (b.team_name == tank.team_name):
                continue
            cnt += 1
            d = dist_xy(tank.x, tank.y, b.x, b.y)
            if d < min_d:
                min_d = d
                min_ang = vector_to_angle_deg(b.x - tank.x, b.y - tank.y)
        if cnt == 0:
            return -1, -1, 0
        return min_d, min_ang, cnt

    def _danger_approx(self, tank, bullets):
        nb_d, nb_ang, nb_cnt = self._nearest_threat_bullet(tank, bullets)
        if nb_cnt == 0 or nb_d < 0:
            return -1, -1
        return nb_d, nb_d / max(BULLET_SPEED, 1e-6)

    def _nearest_supplies(self, tank, supplies):
        nf_d, nf_x, nf_y = -1, -1, -1
        na_d, na_x, na_y = -1, -1, -1

        best_f = 1e18
        best_a = 1e18
        for s in supplies:
            if not s.active:
                continue
            sx, sy = s.rect.centerx, s.rect.centery
            d = dist_xy(tank.x, tank.y, sx, sy)
            if s.type == "fuel":
                if d < best_f:
                    best_f = d
                    nf_d, nf_x, nf_y = d, sx, sy
            else:
                if d < best_a:
                    best_a = d
                    na_d, na_x, na_y = d, sx, sy

        return (nf_d, nf_x, nf_y, na_d, na_x, na_y)

    def _is_near_supply(self, tank, nf_d, na_d):
        best = 1e18
        if nf_d >= 0: best = min(best, nf_d)
        if na_d >= 0: best = min(best, na_d)
        if best >= 1e18:
            return 0
        return 1 if best < 80 else 0

    def build_features(self, tank, enemy, supplies, walls, bullets):
        dx = enemy.x - tank.x
        dy = enemy.y - tank.y
        dist_e = math.hypot(dx, dy)
        ang_to_enemy = vector_to_angle_deg(dx, dy)
        aim_err = angle_diff_deg(tank.turret_angle, ang_to_enemy)
        los_ok = 1 if has_line_of_sight(tank.x, tank.y, enemy.x, enemy.y, walls) else 0

        nf_d, nf_x, nf_y, na_d, na_x, na_y = self._nearest_supplies(tank, supplies)
        is_near_supply = self._is_near_supply(tank, nf_d, na_d)

        fuel_dx = (nf_x - tank.x) if nf_d >= 0 else 0.0
        fuel_dy = (nf_y - tank.y) if nf_d >= 0 else 0.0
        ammo_dx = (na_x - tank.x) if na_d >= 0 else 0.0
        ammo_dy = (na_y - tank.y) if na_d >= 0 else 0.0

        nb_d, nb_ang, nb_cnt = self._nearest_threat_bullet(tank, bullets)
        danger_min, danger_frames = self._danger_approx(tank, bullets)

        can_fire = 1 if tank.ammo > 0 else 0

        feat = {
            "team_id": 0 if tank.team_name == "Green" else 1,

            "self_x": tank.x, "self_y": tank.y, "self_vx": tank.vx(), "self_vy": tank.vy(),
            "self_life": tank.life, "self_fuel": tank.fuel, "self_ammo": tank.ammo,
            "self_fuel_ratio": tank.fuel / MAX_FUEL, "self_ammo_ratio": tank.ammo / MAX_AMMO,

            "enemy_x": enemy.x, "enemy_y": enemy.y, "enemy_vx": enemy.vx(), "enemy_vy": enemy.vy(),
            "dx": dx, "dy": dy, "dist": dist_e,

            "turret_angle": tank.turret_angle,
            "angle_to_enemy": ang_to_enemy,
            "aim_error_deg": aim_err,
            "los_ok": los_ok,

            "nearest_fuel_dist": nf_d if nf_d >= 0 else -1,
            "nearest_fuel_x": nf_x, "nearest_fuel_y": nf_y,
            "nearest_ammo_dist": na_d if na_d >= 0 else -1,
            "nearest_ammo_x": na_x, "nearest_ammo_y": na_y,
            "is_near_supply": is_near_supply,

            "fuel_dx": fuel_dx, "fuel_dy": fuel_dy,
            "ammo_dx": ammo_dx, "ammo_dy": ammo_dy,

            "enemy_bullet_count": nb_cnt,
            "nearest_bullet_dist": nb_d if nb_d >= 0 else -1,
            "nearest_bullet_angle": nb_ang if nb_ang >= 0 else -1,
            "danger_min_dist": danger_min if danger_min >= 0 else -1,
            "danger_frames_to_closest": danger_frames if danger_frames >= 0 else -1,

            "can_fire": can_fire,
        }
        return feat

    def decide(self, tank, enemy, supplies, walls, bullets):
        # 模型缺檔時，回傳 None 讓外部 fallback
        if (not tank.alive) or (not enemy.alive):
            return MOVE_STOP, 0, tank.turret_angle

        # 如果四個模型有任何一個沒載到，就視為不可用
        if not (self.goal_model.ok and self.move_fight_model.ok and self.move_supply_model.ok and self.fire_model.ok):
            return None

        feat = self.build_features(tank, enemy, supplies, walls, bullets)

        turret_cmd = feat["angle_to_enemy"]

        goal = self.goal_model.predict_one(feat)
        if goal is None:
            goal = 0

        if goal == 0:
            move = self.move_fight_model.predict_one(feat)
            if move is None:
                move = MOVE_STOP
        else:
            move = self.move_supply_model.predict_one(feat)
            if move is None:
                move = MOVE_STOP

        fire = 0
        if feat.get("can_fire", 0) == 1:
            pred_fire = self.fire_model.predict_one(feat)
            if pred_fire is not None:
                fire = 1 if pred_fire == 1 else 0

        return move, fire, turret_cmd


# ===================== 規則型 AI（不用模型） =====================
class StrongAI:
    """
    更強的規則AI：
    1) 預判射擊（lead）
    2) 簡易格子路徑 BFS 繞牆
    3) 視線被牆擋住時：會先拆牆（朝最近阻擋點方向射）
    4) 近距離會拉距離/側移
    5) 子彈太近優先閃避
    """
    def __init__(self):
        # 射擊與戰術參數
        self.aim_fire_threshold = 12          # 瞄準誤差 < 8 度就開火
        self.fire_cooldown_frames = 4       # 控制射速（越小越兇）10
        self._fire_cd = 0

        self.evade_bullet_dist = 160         # 子彈距離 < 160 開始閃
        self.kite_dist = 220                 # 太近就拉距離
        self.strafe_dist = 260               # 中距離會側移

        self.low_fuel = 20
        self.low_ammo = 4

        # 導航格子
        self.cell = 40                       # 格子大小（跟牆厚/坦克半徑差不多）
        self.max_bfs_nodes = 2500            # BFS 上限避免太慢

    # ---------- 小工具 ----------
    def _would_collide_wall(self, nx, ny, tank, walls):
        r = pygame.Rect(0, 0, tank.radius * 2, tank.radius * 2)
        r.center = (nx, ny)
        for w in walls:
            if w.is_destroyed():
                continue
            if r.colliderect(w.rect):
                return True
        return False

    def _nearest_supply(self, tank, supplies, stype):
        best = None
        best_d = 1e18
        for s in supplies:
            if not s.active:
                continue
            if s.type != stype:
                continue
            sx, sy = s.rect.centerx, s.rect.centery
            d = dist_xy(tank.x, tank.y, sx, sy)
            if d < best_d:
                best_d = d
                best = (sx, sy, d)
        return best

    def _nearest_threat_bullet(self, tank, bullets):
        best = None
        best_d = 1e18
        for b in bullets:
            if not b.alive:
                continue
            if b.team_name == tank.team_name:
                continue
            d = dist_xy(tank.x, tank.y, b.x, b.y)
            if d < best_d:
                best_d = d
                best = (b.x, b.y, d, b.angle)
        return best

    def _grid_coord(self, x, y):
        return int(x // self.cell), int(y // self.cell)

    def _grid_center(self, gx, gy):
        return gx * self.cell + self.cell * 0.5, gy * self.cell + self.cell * 0.5

    def _build_blocked_grid(self, tank, walls):
        gw = int(WORLD_WIDTH // self.cell) + 1
        gh = int(WORLD_HEIGHT // self.cell) + 1
        blocked = [[False for _ in range(gh)] for _ in range(gw)]

        # 對每個格子中心點，若坦克站那裡會撞牆就視為 blocked
        # 這樣 BFS 走格子時就能「繞牆」
        for gx in range(gw):
            for gy in range(gh):
                cx, cy = self._grid_center(gx, gy)
                if cx < tank.radius or cx > WORLD_WIDTH - tank.radius:
                    blocked[gx][gy] = True
                    continue
                if cy < tank.radius or cy > WORLD_HEIGHT - tank.radius:
                    blocked[gx][gy] = True
                    continue
                if self._would_collide_wall(cx, cy, tank, walls):
                    blocked[gx][gy] = True
        return blocked, gw, gh

    def _bfs_next_step(self, tank, target_x, target_y, walls):
        # BFS 找下一步格子（回傳下一個 waypoint）
        blocked, gw, gh = self._build_blocked_grid(tank, walls)
        sx, sy = self._grid_coord(tank.x, tank.y)
        tx, ty = self._grid_coord(target_x, target_y)

        sx = clamp(sx, 0, gw - 1); sy = clamp(sy, 0, gh - 1)
        tx = clamp(tx, 0, gw - 1); ty = clamp(ty, 0, gh - 1)

        if blocked[sx][sy]:
            # 若起點剛好被判 blocked，就直接不走 BFS
            return None

        from collections import deque
        q = deque()
        q.append((sx, sy))
        prev = { (sx, sy): None }
        visited = 1

        # 4-neighbor
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]

        while q and visited < self.max_bfs_nodes:
            x, y = q.popleft()
            if (x, y) == (tx, ty):
                break
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= gw or ny < 0 or ny >= gh:
                    continue
                if blocked[nx][ny]:
                    continue
                if (nx, ny) in prev:
                    continue
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))
                visited += 1

        if (tx, ty) not in prev:
            return None  # 找不到路

        # 回溯，取「從起點走的第一步」
        cur = (tx, ty)
        while prev[cur] is not None and prev[cur] != (sx, sy):
            cur = prev[cur]
        if prev[cur] is None:
            return None
        wx, wy = self._grid_center(cur[0], cur[1])
        return (wx, wy)

    def _best_move_toward_point(self, tank, px, py, walls):
        candidates = [
            (MOVE_STOP, 0, 0),
            (MOVE_UP, 0, -TANK_SPEED),
            (MOVE_DOWN, 0, TANK_SPEED),
            (MOVE_LEFT, -TANK_SPEED, 0),
            (MOVE_RIGHT, TANK_SPEED, 0),
        ]
        best_cmd = MOVE_STOP
        best_score = 1e18

        for cmd, dx, dy in candidates:
            nx = clamp(tank.x + dx, tank.radius, WORLD_WIDTH - tank.radius)
            ny = clamp(tank.y + dy, tank.radius, WORLD_HEIGHT - tank.radius)
            if self._would_collide_wall(nx, ny, tank, walls):
                continue
            score = dist_xy(nx, ny, px, py)
            if score < best_score:
                best_score = score
                best_cmd = cmd
        return best_cmd

    def _lead_angle(self, shooter, target):
        # 簡單預判：用 target 速度估計未來位置
        tx, ty = target.x, target.y
        tvx, tvy = target.vx(), target.vy()
        dx = tx - shooter.x
        dy = ty - shooter.y
        d = math.hypot(dx, dy)

        # 子彈到達時間 ~ 距離 / 速度
        t = d / max(BULLET_SPEED, 1e-6)

        # 預估目標未來位置（加一點限幅避免爆走）
        fx = tx + clamp(tvx, -TANK_SPEED, TANK_SPEED) * t
        fy = ty + clamp(tvy, -TANK_SPEED, TANK_SPEED) * t

        # 夾回地圖內
        fx = clamp(fx, 0, WORLD_WIDTH)
        fy = clamp(fy, 0, WORLD_HEIGHT)

        return vector_to_angle_deg(fx - shooter.x, fy - shooter.y)

    def decide(self, tank, enemy, supplies, walls, bullets):
        if (not tank.alive) or (not enemy.alive):
            return MOVE_STOP, 0, tank.turret_angle

        # cooldown
        if self._fire_cd > 0:
            self._fire_cd -= 1

        # 1) 子彈閃避（優先）
        # 1) 子彈閃避（優先）— 改成「邊閃邊射」
        nb = self._nearest_threat_bullet(tank, bullets)
        if nb is not None:
            bx, by, bd, b_ang = nb
            if bd < self.evade_bullet_dist:
                # 往子彈->坦克向量的垂直方向閃
                ux, uy = safe_unit(tank.x - bx, tank.y - by)
                px1, py1 = -uy, ux
                px2, py2 = uy, -ux

                t1x = clamp(tank.x + px1 * TANK_SPEED, tank.radius, WORLD_WIDTH - tank.radius)
                t1y = clamp(tank.y + py1 * TANK_SPEED, tank.radius, WORLD_HEIGHT - tank.radius)
                t2x = clamp(tank.x + px2 * TANK_SPEED, tank.radius, WORLD_WIDTH - tank.radius)
                t2y = clamp(tank.y + py2 * TANK_SPEED, tank.radius, WORLD_HEIGHT - tank.radius)

                ok1 = not self._would_collide_wall(t1x, t1y, tank, walls)
                ok2 = not self._would_collide_wall(t2x, t2y, tank, walls)

                # 邊閃邊用預判瞄準敵人
                turret_cmd = self._lead_angle(tank, enemy)

                # 決定要往哪側閃
                if ok1 or ok2:
                    d1 = dist_xy(t1x, t1y, bx, by) if ok1 else -1
                    d2 = dist_xy(t2x, t2y, bx, by) if ok2 else -1
                    if d1 >= d2:
                        move_cmd = self._best_move_toward_point(tank, t1x, t1y, walls)
                    else:
                        move_cmd = self._best_move_toward_point(tank, t2x, t2y, walls)
                else:
                    move_cmd = MOVE_STOP

                # ====== 這裡是新增：閃避時也允許射擊 ======
                fire_cmd = 0
                if tank.ammo > 0 and self._fire_cd == 0:
                    # 閃避時放寬一點角度門檻（比較容易射）
                    aim_err = angle_diff_deg(tank.turret_angle, turret_cmd)
                    if aim_err <= max(self.aim_fire_threshold, 18):
                        fire_cmd = 1
                        self._fire_cd = self.fire_cooldown_frames

                return move_cmd, fire_cmd, turret_cmd


        # 2) 砲塔：預判瞄準敵人（比死瞄準強）
        turret_cmd = self._lead_angle(tank, enemy)

        # 3) 決定戰略目標（補包 or 壓制）
        target_x, target_y = enemy.x, enemy.y
        want_supply = False

        if tank.fuel < self.low_fuel:
            s = self._nearest_supply(tank, supplies, "fuel")
            if s is not None:
                target_x, target_y, _ = s
                want_supply = True

        if tank.ammo < self.low_ammo:
            s = self._nearest_supply(tank, supplies, "ammo")
            if s is not None:
                target_x, target_y, _ = s
                want_supply = True

        # 4) 走位（繞牆 BFS waypoint）
        wp = self._bfs_next_step(tank, target_x, target_y, walls)
        if wp is None:
            move_cmd = self._best_move_toward_point(tank, target_x, target_y, walls)
        else:
            move_cmd = self._best_move_toward_point(tank, wp[0], wp[1], walls)

        # 5) 拉距離/側移（打架更像人）
        d_enemy = dist_xy(tank.x, tank.y, enemy.x, enemy.y)
        if not want_supply:
            if d_enemy < self.kite_dist:
                # 後退（往遠離敵人的方向）
                ax, ay = safe_unit(tank.x - enemy.x, tank.y - enemy.y)
                px = tank.x + ax * TANK_SPEED
                py = tank.y + ay * TANK_SPEED
                move_cmd = self._best_move_toward_point(tank, px, py, walls)
            elif d_enemy < self.strafe_dist:
                # 側移（垂直於敵->我向量）
                ax, ay = safe_unit(enemy.x - tank.x, enemy.y - tank.y)
                sx, sy = -ay, ax
                if random.random() < 0.5:
                    sx, sy = -sx, -sy
                px = tank.x + sx * TANK_SPEED
                py = tank.y + sy * TANK_SPEED
                move_cmd = self._best_move_toward_point(tank, px, py, walls)

        # 6) 開火策略：能看到就打；看不到就拆牆開路
        fire_cmd = 0
        if tank.ammo > 0 and self._fire_cd == 0:
            los_ok = has_line_of_sight(tank.x, tank.y, enemy.x, enemy.y, walls)
            aim_err = angle_diff_deg(tank.turret_angle, turret_cmd)

            if los_ok and aim_err <= self.aim_fire_threshold:
                fire_cmd = 1
                self._fire_cd = self.fire_cooldown_frames
            else:
                # 看不到敵人：找「第一個擋路的牆」並嘗試拆牆（非常關鍵，AI會變兇）
                hit, hx, hy, hd = first_wall_hit_point(tank.x, tank.y, enemy.x, enemy.y, walls)
                if hit and hd > 0 and hd < 500:
                    # 砲塔轉向牆的阻擋點
                    turret_cmd = vector_to_angle_deg(hx - tank.x, hy - tank.y)
                    aim_err2 = angle_diff_deg(tank.turret_angle, turret_cmd)
                    if aim_err2 <= 12:  # 拆牆容忍度可以大一點
                        fire_cmd = 1
                        self._fire_cd = self.fire_cooldown_frames

        return move_cmd, fire_cmd, turret_cmd


# ===================== 遊戲主程式 =====================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tank - Model vs RuleBased AI (Random Respawn)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)
    big_font = pygame.font.SysFont(None, 48)

    camera_x = 0
    camera_y = 0
    CAMERA_SPEED = 15

    team_scores = {"Green": 0, "Blue": 0}

    # ---- 牆 ----
    walls = []
    BORDER_THICKNESS = 40
    walls.append(Wall(0, 0, WORLD_WIDTH, BORDER_THICKNESS))
    walls.append(Wall(0, WORLD_HEIGHT - BORDER_THICKNESS, WORLD_WIDTH, BORDER_THICKNESS))
    walls.append(Wall(0, 0, BORDER_THICKNESS, WORLD_HEIGHT))
    walls.append(Wall(WORLD_WIDTH - BORDER_THICKNESS, 0, BORDER_THICKNESS, WORLD_HEIGHT))

    # 內部牆
        # 內部牆（置中生成）
    cx = WORLD_WIDTH // 2
    cy = WORLD_HEIGHT // 2

    # 參數：你可以自己調
    H_COUNT = 5          # 橫向牆數量
    V_COUNT = 3          # 縱向牆數量
    GAP = 200            # 牆與牆中心間距（越大越散）
    H_W, H_H = 120, 40   # 橫向牆尺寸
    V_W, V_H = 40, 120   # 縱向牆尺寸

    # 橫向牆：以 (cx, cy) 為中心左右展開
    start_x = cx - (H_COUNT - 1) * GAP // 2
    y = cy - H_H // 2
    for i in range(H_COUNT):
        x = start_x + i * GAP
        walls.append(Wall(x, y, H_W, H_H))

    # 縱向牆：以 (cx, cy) 為中心上下展開
    start_y = cy - (V_COUNT - 1) * GAP // 2
    x = cx - V_W // 2
    for i in range(V_COUNT):
        y = start_y + i * GAP
        walls.append(Wall(x, y, V_W, V_H))


    # ---- 補給 ----
    supplies = []
    for _ in range(3):
        x = random.randint(0, WORLD_WIDTH - 30)
        y = random.randint(0, WORLD_HEIGHT - 30)
        supplies.append(Supply("fuel", x, y))
    for _ in range(3):
        x = random.randint(0, WORLD_WIDTH - 30)
        y = random.randint(0, WORLD_HEIGHT - 30)
        supplies.append(Supply("ammo", x, y))

    bullets = []

    # ---- 坦克 ----
    tank_model = Tank(60, 60, GREEN, "Green")  # 模型方
    tank_ai    = Tank(WORLD_WIDTH - 60, WORLD_HEIGHT - 60, BLUE, "Blue")  # 規則AI方
    tanks = [tank_model, tank_ai]

    # ---- 初始隨機出生 ----
    x1, y1 = random_respawn_pos(walls, supplies, [tank_ai], tank_model.radius)
    tank_model.respawn(x1, y1)
    x2, y2 = random_respawn_pos(walls, supplies, [tank_model], tank_ai.radius)
    tank_ai.respawn(x2, y2)

    # ---- AI 載入 ----
    model_ai = ModelTankAI()
    rule_ai  = StrongAI()

    start_ticks = pygame.time.get_ticks()
    game_over = False
    winner_text = ""

    def apply_move_cmd(tank, cmd):
        tank.move_forward  = (cmd == MOVE_UP)
        tank.move_backward = (cmd == MOVE_DOWN)
        tank.move_left     = (cmd == MOVE_LEFT)
        tank.move_right    = (cmd == MOVE_RIGHT)

    def try_fire(tank, fire_cmd):
        if fire_cmd != 1:
            return
        if tank.ammo <= 0 or (not tank.alive) or game_over:
            return
        tx, ty = angle_to_vector(tank.turret_angle)
        bx = tank.x + tx * 30
        by = tank.y + ty * 30
        bullets.append(Bullet(bx, by, tank.turret_angle, tank.team_name))
        tank.ammo -= 1

    while True:
        clock.tick(FPS)

        elapsed_ms = pygame.time.get_ticks() - start_ticks
        remain_sec = max(0, GAME_TIME_SECONDS - int(elapsed_ms / 1000))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_i: camera_y -= CAMERA_SPEED
                if event.key == pygame.K_k: camera_y += CAMERA_SPEED
                if event.key == pygame.K_j: camera_x -= CAMERA_SPEED
                if event.key == pygame.K_l: camera_x += CAMERA_SPEED

        if not game_over:
            # Green（模型）
            out = model_ai.decide(tank_model, tank_ai, supplies, walls, bullets)
            if out is None:
                # 若模型沒載到，Green 也用規則AI fallback，避免你直接不能玩
                m1, f1, t1 = rule_ai.decide(tank_model, tank_ai, supplies, walls, bullets)
            else:
                m1, f1, t1 = out

            # Blue（規則AI）
            m2, f2, t2 = rule_ai.decide(tank_ai, tank_model, supplies, walls, bullets)

            tank_model.turret_angle = t1 % 360
            tank_ai.turret_angle    = t2 % 360

            apply_move_cmd(tank_model, m1)
            apply_move_cmd(tank_ai, m2)

            try_fire(tank_model, f1)
            try_fire(tank_ai, f2)

            for tank in tanks:
                tank.update(walls)

            for b in bullets:
                b.update()
            bullets = [b for b in bullets if b.alive]

            for s in supplies:
                s.update()

            # ---- 子彈碰撞 ----
            for b in bullets:
                if not b.alive:
                    continue

                # hit tank
                for tk in tanks:
                    if not tk.alive:
                        continue
                    if tk.team_name == b.team_name:
                        continue
                    if dist_xy(b.x, b.y, tk.x, tk.y) <= b.radius + tk.radius:
                        b.alive = False
                        killed = tk.hit_by_bullet()
                        if killed:
                            team_scores[b.team_name] += 20
                            other = [t for t in tanks if t is not tk]
                            rx, ry = random_respawn_pos(walls, supplies, other, tk.radius)
                            tk.respawn(rx, ry)
                        break

                # hit wall
                if b.alive:
                    for w in walls:
                        if w.is_destroyed():
                            continue
                        if w.rect.collidepoint(b.x, b.y):
                            b.alive = False
                            team_scores[b.team_name] += 1
                            w.hit()
                            if w.is_destroyed():
                                team_scores[b.team_name] += 5
                            break

                # hit supply
                if b.alive:
                    for s in supplies:
                        if s.active and s.rect.collidepoint(b.x, b.y):
                            b.alive = False
                            s.consume()
                            break

            bullets = [b for b in bullets if b.alive]

            # ---- 撿補包 ----
            for tk in tanks:
                if not tk.alive:
                    continue
                t_rect = pygame.Rect(0, 0, tk.radius * 2, tk.radius * 2)
                t_rect.center = (tk.x, tk.y)
                for s in supplies:
                    if s.active and t_rect.colliderect(s.rect):
                        if s.type == "fuel":
                            tk.fuel = min(MAX_FUEL, tk.fuel + FUEL_SUPPLY_AMOUNT)
                        else:
                            tk.ammo = min(MAX_AMMO, tk.ammo + AMMO_SUPPLY_AMOUNT)
                        s.consume()

            if remain_sec <= 0:
                game_over = True
                if team_scores["Green"] > team_scores["Blue"]:
                    winner_text = "Green(model) WIN"
                elif team_scores["Blue"] > team_scores["Green"]:
                    winner_text = "Blue(AI) WIN"
                else:
                    winner_text = "DRAW"

        # ========================= 繪製 =========================
        screen.fill((30, 30, 30))
        pygame.draw.rect(screen, DARKGREY, (-camera_x, -camera_y, WORLD_WIDTH, WORLD_HEIGHT), 4)

        for wall in walls:
            wall.draw(screen, (camera_x, camera_y))

        for s in supplies:
            s.draw(screen, (camera_x, camera_y))

        for tank in tanks:
            tank.draw(screen, (camera_x, camera_y))

        for b in bullets:
            b.draw(screen, (camera_x, camera_y))

        ui_y = 10
        screen.blit(font.render(f"Green(model) score: {team_scores['Green']}", True, GREEN), (10, ui_y)); ui_y += 28
        screen.blit(font.render(f"Blue(AI) score: {team_scores['Blue']}", True, BLUE), (10, ui_y)); ui_y += 28
        screen.blit(font.render(f"time: {remain_sec} s", True, WHITE), (10, ui_y)); ui_y += 28

        p1_info = font.render(f"Green: HP {tank_model.life} / Fuel {int(tank_model.fuel)} / Ammo {tank_model.ammo}", True, GREEN)
        p2_info = font.render(f"Blue : HP {tank_ai.life} / Fuel {int(tank_ai.fuel)} / Ammo {tank_ai.ammo}", True, BLUE)
        screen.blit(p1_info, (10, ui_y)); ui_y += 28
        screen.blit(p2_info, (10, ui_y)); ui_y += 28

        cam_info = font.render("I/J/K/L move | ESC leave", True, WHITE)
        screen.blit(cam_info, (10, SCREEN_HEIGHT - 30))

        if game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))
            wt = big_font.render(winner_text, True, WHITE)
            screen.blit(wt, wt.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))

        pygame.display.flip()


if __name__ == "__main__":
    main()
