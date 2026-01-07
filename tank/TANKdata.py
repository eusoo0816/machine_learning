import pygame
import random
import math
import os
import csv
from datetime import datetime

# ===================== 基本設定（不改遊戲參數） =====================
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
GREEN = (0, 200, 0)
BLUE  = (0, 128, 255)
RED   = (255, 0, 0)
YELLOW = (255, 255, 0)
GREY = (120, 120, 120)
DARKGREY = (50, 50, 50)

CSV_DIR = r"C:\Users\kai\Desktop\Machinelearning\TANK\tree\treedata"

# ===================== 小工具 =====================
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

def safe_unit(ax, ay, eps=1e-9):
    n = math.hypot(ax, ay)
    if n < eps:
        return 0.0, 0.0
    return ax / n, ay / n

def dot(ax, ay, bx, by):
    return ax*bx + ay*by

def dist_xy(x1, y1, x2, y2):
    return math.hypot(x2-x1, y2-y1)

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
            if not w.is_destroyed() and w.rect.collidepoint(px, py):
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

# ===================== 類別 =====================
class Tank:
    def __init__(self, x, y, color, team_name):
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y

        self.color = color
        self.team_name = team_name

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

    def update(self):
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

        # 這是遊戲機制：移動會耗油（你這支是蒐集程式，不建議改）
        if dx != 0 or dy != 0:
            self.fuel = max(0, self.fuel - 0.1)

        self.x += dx
        self.y += dy

        self.x = clamp(self.x, self.radius, WORLD_WIDTH  - self.radius)
        self.y = clamp(self.y, self.radius, WORLD_HEIGHT - self.radius)

    def draw(self, surface, camera_offset):
        if not self.alive:
            return
        cx = int(self.x - camera_offset[0])
        cy = int(self.y - camera_offset[1])

        body_surf = pygame.Surface((40, 30), pygame.SRCALPHA)
        pygame.draw.rect(body_surf, self.color, (0, 0, 40, 30))
        rect = body_surf.get_rect(center=(cx, cy))
        surface.blit(body_surf, rect.topleft)

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
        self.type = supply_type  # "fuel" or "ammo"
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
        t_surf = font.render(text, True, (255,255,255))
        t_rect = t_surf.get_rect(center=(x + self.rect.width // 2, y + self.rect.height // 2))
        surface.blit(t_surf, t_rect)

# ===================== AI（蒐集專用：強閃避 + 強補包 + 連發） =====================
MOVE_STOP, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT = 0, 1, 2, 3, 4

# ===================== AI（蒐集專用：強閃避 + 強補包 + 永遠開火） =====================
MOVE_STOP, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT = 0, 1, 2, 3, 4

class TankAI:
    """
    action_goal:
      0=fight
      1=ammo
      2=fuel
      3=dodge
    """

    def __init__(self, name, epsilon=0.01):
        self.name = name
        self.epsilon = epsilon

        self.prev_pos = None
        self.stuck_count = 0
        self.unstuck_hold = 0
        self.unstuck_cmd = MOVE_STOP

        # 永遠開火：不需要 cooldown，但保留欄位避免你其它程式有依賴
        self.fire_cooldown = 0

        # 供「在補給都充足時」也會輪流吃
        self.force_supply_toggle = 0  # 0 -> ammo, 1 -> fuel

        # dodge 記憶
        self.dodge_hold = 0
        self.last_dodge_cmd = MOVE_STOP

    # ---------------- 小工具 ----------------
    def _check_stuck(self, tank):
        if self.prev_pos is None:
            self.prev_pos = (tank.x, tank.y)
            self.stuck_count = 0
            return

        moved = math.hypot(tank.x - self.prev_pos[0], tank.y - self.prev_pos[1])
        self.prev_pos = (tank.x, tank.y)

        if moved < 1.0:
            self.stuck_count += 1
        else:
            self.stuck_count = max(0, self.stuck_count - 1)

        if self.stuck_count >= 14 and self.unstuck_hold <= 0:
            self.unstuck_hold = 14
            self.unstuck_cmd = random.choice([MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT])
            self.stuck_count = 0

    def _move_toward_point(self, tank, tx, ty):
        dx = tx - tank.x
        dy = ty - tank.y
        if abs(dx) > abs(dy):
            return MOVE_RIGHT if dx > 0 else MOVE_LEFT
        else:
            return MOVE_DOWN if dy > 0 else MOVE_UP

    def _nearest_supply(self, tank, supplies, prefer_type=None):
        best = None
        best_d = 1e18
        for s in supplies:
            if not s.active:
                continue
            if prefer_type is not None and s.type != prefer_type:
                continue
            sx, sy = s.rect.centerx, s.rect.centery
            d = dist_xy(tank.x, tank.y, sx, sy)
            if d < best_d:
                best_d = d
                best = (s.type, sx, sy, d)
        return best

    def _nearest_any_supply(self, tank, supplies):
        return self._nearest_supply(tank, supplies, prefer_type=None)

    def _nearest_threat_bullet(self, tank, bullets):
        min_d = 1e9
        min_ang = 0.0
        cnt = 0
        for b in bullets:
            if not b.alive or b.team_name == tank.team_name:
                continue
            cnt += 1
            d = dist_xy(tank.x, tank.y, b.x, b.y)
            if d < min_d:
                min_d = d
                min_ang = vector_to_angle_deg(b.x - tank.x, b.y - tank.y)
        if cnt == 0:
            return -1, -1, 0
        return min_d, min_ang, cnt

    def _most_dangerous_bullet(self, tank, bullets, lookahead_frames=60, danger_dist=110.0):
        """
        加強版：看更遠(lookahead_frames=60)，危險距離更大(danger_dist=110)
        """
        best = None
        best_score = None

        for b in bullets:
            if not b.alive or b.team_name == tank.team_name:
                continue

            bvx, bvy = angle_to_vector(b.angle)
            bvx *= BULLET_SPEED
            bvy *= BULLET_SPEED

            rx = tank.x - b.x
            ry = tank.y - b.y

            proj = dot(rx, ry, bvx, bvy)
            if proj <= 0:
                continue

            v2 = bvx*bvx + bvy*bvy
            if v2 < 1e-9:
                continue
            v = math.sqrt(v2)

            distance_along = proj / v
            frames_to_closest = distance_along / BULLET_SPEED
            if frames_to_closest > lookahead_frames:
                continue

            cx = b.x + (bvx / v) * distance_along
            cy = b.y + (bvy / v) * distance_along
            min_dist = math.hypot(tank.x - cx, tank.y - cy)

            hit_radius = tank.radius + b.radius + 18.0
            if min_dist > max(danger_dist, hit_radius):
                continue

            # 分數：越快撞到 & 越近 越危險
            score = (lookahead_frames - frames_to_closest) * 12.0 + (140.0 - min_dist) * 6.5

            if best is None or score > best_score:
                best = {
                    "bullet": b,
                    "frames_to_closest": frames_to_closest,
                    "min_dist": min_dist,
                    "bvx": bvx,
                    "bvy": bvy,
                }
                best_score = score

        return best

    def _dodge_move_perp(self, tank, danger_info):
        bvx, bvy = danger_info["bvx"], danger_info["bvy"]
        ux, uy = safe_unit(bvx, bvy)

        # 子彈方向的法向量
        nx, ny = -uy, ux

        b = danger_info["bullet"]
        rx = tank.x - b.x
        ry = tank.y - b.y
        side = dot(rx, ry, nx, ny)

        # 偶爾反向，避免卡牆/可預測
        if random.random() < 0.10:
            side = -side

        mx = nx if side >= 0 else -nx
        my = ny if side >= 0 else -ny

        if abs(mx) > abs(my):
            return MOVE_RIGHT if mx > 0 else MOVE_LEFT
        else:
            return MOVE_DOWN if my > 0 else MOVE_UP

    # ---------------- 核心策略 ----------------
    def _fire_policy_always(self, tank, los_ok):
        """
        永遠開火：只要有子彈就一直回傳 1
        （los_ok 仍會被記錄到 CSV，但不再影響是否開火）
        """
        if tank.ammo <= 0:
            return 0
        return 1

    def _choose_supply_prefer(self, tank):
        """
        讓 AI 真的「一直吃補包」而且兩種都吃：
        - ammo 太少 -> 優先 ammo
        - fuel 太少 -> 優先 fuel
        - 都夠 -> 依 toggle 輪流吃，避免停在某區域
        """
        ammo_ratio = tank.ammo / MAX_AMMO
        fuel_ratio = tank.fuel / MAX_FUEL

        if ammo_ratio < 0.65:
            return "ammo"
        if fuel_ratio < 0.65:
            return "fuel"

        # 都很夠：輪流
        return "ammo" if self.force_supply_toggle == 0 else "fuel"

    def decide(self, tank, enemy, supplies, walls, bullets, frame_id=0):
        if (not tank.alive) or (not enemy.alive):
            return MOVE_STOP, 0, tank.turret_angle, -1, -1, -1, 0, -1, -1, 0, "none", "none", -1, -1, -1, 0

        dxE = enemy.x - tank.x
        dyE = enemy.y - tank.y
        ang_to_enemy = vector_to_angle_deg(dxE, dyE)
        los_ok = 1 if has_line_of_sight(tank.x, tank.y, enemy.x, enemy.y, walls) else 0

        nb_d, nb_ang, nb_cnt = self._nearest_threat_bullet(tank, bullets)

        self._check_stuck(tank)

        # ------------------------------------------------------------
        # 1) DODGE 最優先（更強的危險偵測）
        # ------------------------------------------------------------
        danger = self._most_dangerous_bullet(tank, bullets, lookahead_frames=60, danger_dist=110.0)
        if danger is not None:
            danger_min_dist = danger["min_dist"]
            danger_frames = danger["frames_to_closest"]

            if self.dodge_hold <= 0:
                self.last_dodge_cmd = self._dodge_move_perp(tank, danger)
                # 躲久一點，真的看起來像在閃
                self.dodge_hold = 14 if danger_frames < 12 else 10
            self.dodge_hold -= 1

            turret_cmd = ang_to_enemy
            aim_err = angle_diff_deg(tank.turret_angle, turret_cmd)
            fire_cmd = self._fire_policy_always(tank, los_ok)

            return (
                self.last_dodge_cmd, fire_cmd, turret_cmd,
                danger_min_dist, danger_frames,
                aim_err, los_ok,
                nb_d, nb_ang, nb_cnt,
                "none",
                "enemy", -1, -1, -1,
                3  # action_goal = dodge
            )

        danger_min_dist, danger_frames = -1, -1

        # ------------------------------------------------------------
        # 2) 平常狀態：幾乎永遠追補給（一直吃彈藥包/補包）
        # ------------------------------------------------------------
        prefer = self._choose_supply_prefer(tank)
        picked = self._nearest_supply(tank, supplies, prefer_type=prefer)
        if picked is None:
            picked = self._nearest_any_supply(tank, supplies)

        action_goal = 0
        target_supply_type = "none"

        if picked is not None:
            stype, sx, sy, sd = picked
            target_supply_type = stype
            action_goal = 1 if stype == "ammo" else 2
            move_cmd = self._move_toward_point(tank, sx, sy)

            # 當靠近目標補給時，輪換一次，讓它下一輪去吃另一種
            if sd < 120:
                self.force_supply_toggle = 1 - self.force_supply_toggle
        else:
            # 沒有任何補給可吃：才追敵
            move_cmd = self._move_toward_point(tank, enemy.x, enemy.y)
            action_goal = 0
            target_supply_type = "none"

        # 解卡
        if self.unstuck_hold > 0:
            self.unstuck_hold -= 1
            move_cmd = self.unstuck_cmd

        # ------------------------------------------------------------
        # 3) turret：永遠瞄敵（你原本的打牆邏輯可保留）
        #     - 為了「永遠開火」更有效：LOS 不通就先打牆穿洞
        # ------------------------------------------------------------
        target_type = "enemy"
        wall_hit_x, wall_hit_y, wall_hit_dist = -1, -1, -1

        turret_cmd = ang_to_enemy
        if los_ok == 0:
            hit, hx, hy, hdist = first_wall_hit_point(tank.x, tank.y, enemy.x, enemy.y, walls, step=10)
            if hit:
                turret_cmd = vector_to_angle_deg(hx - tank.x, hy - tank.y)
                target_type = "wall"
                wall_hit_x, wall_hit_y, wall_hit_dist = hx, hy, hdist

        aim_err = angle_diff_deg(tank.turret_angle, turret_cmd)

        # 永遠開火
        fire_cmd = self._fire_policy_always(tank, los_ok)

        # 少量探索（保留你的資料多樣性），但降低到非常小
        if random.random() < (self.epsilon * 0.5):
            move_cmd = random.choice([MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT])

        return (
            move_cmd, fire_cmd, turret_cmd,
            danger_min_dist, danger_frames,
            aim_err, los_ok,
            nb_d, nb_ang, nb_cnt,
            target_supply_type,
            target_type, wall_hit_x, wall_hit_y, wall_hit_dist,
            action_goal
        )

# ===================== DataLogger（含 dodge goal=3） =====================
class DataLogger:
    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(out_dir, f"tank_dataset_v6_goal4_withDodge_{ts}.csv")
        self.f = open(self.path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)

        self.w.writerow([
            "frame", "team",

            "self_x", "self_y", "self_vx", "self_vy",
            "self_life", "self_fuel", "self_ammo",
            "self_fuel_ratio", "self_ammo_ratio",

            "enemy_x", "enemy_y", "enemy_vx", "enemy_vy",
            "dx", "dy", "dist",

            "turret_angle", "angle_to_enemy", "aim_error_deg",
            "los_ok",

            "target_type",
            "wall_hit_x", "wall_hit_y", "wall_hit_dist",

            "nearest_fuel_dist", "nearest_fuel_x", "nearest_fuel_y",
            "nearest_ammo_dist", "nearest_ammo_x", "nearest_ammo_y",
            "target_supply_type",
            "is_near_supply",

            "fuel_dx", "fuel_dy",
            "ammo_dx", "ammo_dy",

            "enemy_bullet_count",
            "nearest_bullet_dist", "nearest_bullet_angle",
            "danger_min_dist", "danger_frames_to_closest",

            "can_fire",
            "action_goal",         # 0 fight / 1 ammo / 2 fuel / 3 dodge
            "sample_weight",
            "action_move", "action_fire", "turret_angle_cmd",
        ])

    def close(self):
        try:
            self.f.flush()
        finally:
            self.f.close()

    def log(self, frame_id, tank, enemy, supplies,
            action_goal, action_move, action_fire, turret_cmd,
            danger_min_dist, danger_frames,
            aim_error_deg_val, los_ok,
            nearest_bullet_dist, nearest_bullet_angle, enemy_bullet_count,
            target_supply_type,
            target_type, wall_hit_x, wall_hit_y, wall_hit_dist):

        if (not tank.alive) or (not enemy.alive):
            return

        dx = enemy.x - tank.x
        dy = enemy.y - tank.y
        d = math.hypot(dx, dy)
        angle_to_enemy = vector_to_angle_deg(dx, dy)

        nf_d, nf_x, nf_y = 1e9, -1, -1
        na_d, na_x, na_y = 1e9, -1, -1

        for s in supplies:
            if not s.active:
                continue
            sx, sy = s.rect.centerx, s.rect.centery
            sd = dist_xy(tank.x, tank.y, sx, sy)
            if s.type == "fuel":
                if sd < nf_d:
                    nf_d, nf_x, nf_y = sd, sx, sy
            else:
                if sd < na_d:
                    na_d, na_x, na_y = sd, sx, sy

        if nf_d >= 1e9: nf_d = -1
        if na_d >= 1e9: na_d = -1

        best_supply_dist = 1e9
        if nf_d >= 0: best_supply_dist = min(best_supply_dist, nf_d)
        if na_d >= 0: best_supply_dist = min(best_supply_dist, na_d)
        is_near_supply = 1 if best_supply_dist < 80 else 0

        can_fire = 1 if tank.ammo > 0 else 0

        fuel_dx = (nf_x - tank.x) if nf_d >= 0 else 0.0
        fuel_dy = (nf_y - tank.y) if nf_d >= 0 else 0.0
        ammo_dx = (na_x - tank.x) if na_d >= 0 else 0.0
        ammo_dy = (na_y - tank.y) if na_d >= 0 else 0.0

        # -----------------------------
        # sample_weight：偏壓（含 dodge）
        # -----------------------------
        w = 1.0
        if int(action_goal) in (1, 2):   # supply
            w *= 6.0
        if int(action_goal) == 3:        # dodge：也要讓樹重視，不然會被淹掉
            w *= 8.0
        if int(action_fire) == 1:
            w *= 3.0
        if can_fire == 0:
            w *= 0.15
        if is_near_supply == 1:
            w *= 1.5

        w = min(max(w, 0.05), 40.0)

        self.w.writerow([
            frame_id, tank.team_name,

            round(tank.x, 2), round(tank.y, 2), round(tank.vx(), 2), round(tank.vy(), 2),
            tank.life, round(tank.fuel, 2), tank.ammo,
            round(tank.fuel / MAX_FUEL, 4), round(tank.ammo / MAX_AMMO, 4),

            round(enemy.x, 2), round(enemy.y, 2), round(enemy.vx(), 2), round(enemy.vy(), 2),
            round(dx, 2), round(dy, 2), round(d, 2),

            round(tank.turret_angle, 2), round(angle_to_enemy, 2), round(aim_error_deg_val, 2),
            int(los_ok),

            target_type,
            round(wall_hit_x, 2) if wall_hit_x >= 0 else -1,
            round(wall_hit_y, 2) if wall_hit_y >= 0 else -1,
            round(wall_hit_dist, 2) if wall_hit_dist >= 0 else -1,

            round(nf_d, 2) if nf_d >= 0 else -1, nf_x, nf_y,
            round(na_d, 2) if na_d >= 0 else -1, na_x, na_y,
            target_supply_type,
            is_near_supply,

            round(fuel_dx, 2), round(fuel_dy, 2),
            round(ammo_dx, 2), round(ammo_dy, 2),

            int(enemy_bullet_count),
            round(nearest_bullet_dist, 2) if nearest_bullet_dist >= 0 else -1,
            round(nearest_bullet_angle, 2) if nearest_bullet_angle >= 0 else -1,
            round(danger_min_dist, 2) if danger_min_dist >= 0 else -1,
            round(danger_frames, 2) if danger_frames >= 0 else -1,

            can_fire,
            int(action_goal),
            float(w),
            int(action_move), int(action_fire), round(turret_cmd, 2),
        ])

# ===================== 主程式（AI vs AI + CSV） =====================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tank - Data Collect v6 (Goal4: fight/ammo/fuel/dodge)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 26)
    big_font = pygame.font.SysFont(None, 46)

    camera_x, camera_y = 0, 0
    team_scores = {"Green": 0, "Blue": 0}

    spawn_green = (150, 150)
    spawn_blue  = (SCREEN_WIDTH - 150, SCREEN_HEIGHT - 150)

    tank1 = Tank(*spawn_green, GREEN, "Green")
    tank2 = Tank(*spawn_blue, BLUE, "Blue")
    tanks = [tank1, tank2]

    walls = []
    BORDER = 40
    walls.append(Wall(0, 0, WORLD_WIDTH, BORDER))
    walls.append(Wall(0, WORLD_HEIGHT - BORDER, WORLD_WIDTH, BORDER))
    walls.append(Wall(0, 0, BORDER, WORLD_HEIGHT))
    walls.append(Wall(WORLD_WIDTH - BORDER, 0, BORDER, WORLD_HEIGHT))

    for i in range(4):
        walls.append(Wall(220 + i * 160, 520, 110, 35))
    for i in range(3):
        walls.append(Wall(520, 160 + i * 160, 35, 110))

    supplies = []
    for _ in range(3):
        supplies.append(Supply("fuel", random.randint(0, WORLD_WIDTH - 30), random.randint(0, WORLD_HEIGHT - 30)))
    for _ in range(3):
        supplies.append(Supply("ammo", random.randint(0, WORLD_WIDTH - 30), random.randint(0, WORLD_HEIGHT - 30)))

    bullets = []

    ai1 = TankAI("AI_Green", epsilon=0.02)
    ai2 = TankAI("AI_Blue",  epsilon=0.02)

    logger = DataLogger(CSV_DIR)
    print(f"[INFO] CSV 輸出路徑: {logger.path}")

    start_ticks = pygame.time.get_ticks()
    frame_id = 0
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
        if tank.ammo <= 0 or (not tank.alive):
            return
        tx, ty = angle_to_vector(tank.turret_angle)
        bx = tank.x + tx * 30
        by = tank.y + ty * 30
        bullets.append(Bullet(bx, by, tank.turret_angle, tank.team_name))
        tank.ammo -= 1

    try:
        while True:
            clock.tick(FPS)
            frame_id += 1

            elapsed_ms = pygame.time.get_ticks() - start_ticks
            remain_sec = max(0, GAME_TIME_SECONDS - int(elapsed_ms / 1000))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

            if not game_over:
                (m1, f1, t1, dmin1, dfrm1, aim1, los1, nb_d1, nb_a1, nb_cnt1,
                 tgt1, tgtType1, whx1, why1, whd1, goal1) = ai1.decide(tank1, tank2, supplies, walls, bullets, frame_id=frame_id)

                (m2, f2, t2, dmin2, dfrm2, aim2, los2, nb_d2, nb_a2, nb_cnt2,
                 tgt2, tgtType2, whx2, why2, whd2, goal2) = ai2.decide(tank2, tank1, supplies, walls, bullets, frame_id=frame_id)

                logger.log(frame_id, tank1, tank2, supplies, goal1, m1, f1, t1, dmin1, dfrm1, aim1, los1, nb_d1, nb_a1, nb_cnt1, tgt1, tgtType1, whx1, why1, whd1)
                logger.log(frame_id, tank2, tank1, supplies, goal2, m2, f2, t2, dmin2, dfrm2, aim2, los2, nb_d2, nb_a2, nb_cnt2, tgt2, tgtType2, whx2, why2, whd2)

                tank1.turret_angle = t1 % 360
                tank2.turret_angle = t2 % 360

                apply_move_cmd(tank1, m1)
                apply_move_cmd(tank2, m2)

                try_fire(tank1, f1)
                try_fire(tank2, f2)

                for tk in tanks:
                    tk.update()

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
                                if tk.team_name == "Green":
                                    tk.respawn(*spawn_green)
                                else:
                                    tk.respawn(*spawn_blue)
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
                    t_rect = pygame.Rect(0, 0, tk.radius*2, tk.radius*2)
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
                        winner_text = "時間到！Green Team 勝利！"
                    elif team_scores["Blue"] > team_scores["Green"]:
                        winner_text = "時間到！Blue Team 勝利！"
                    else:
                        winner_text = "時間到！雙方平手！"

            # ===================== Draw =====================
            screen.fill((30,30,30))
            pygame.draw.rect(screen, DARKGREY, (-camera_x, -camera_y, WORLD_WIDTH, WORLD_HEIGHT), 4)

            for w in walls:
                w.draw(screen, (camera_x, camera_y))
            for s in supplies:
                s.draw(screen, (camera_x, camera_y))
            for tk in tanks:
                tk.draw(screen, (camera_x, camera_y))
            for b in bullets:
                b.draw(screen, (camera_x, camera_y))

            ui_y = 10
            screen.blit(font.render(f"Green: {team_scores['Green']}", True, GREEN), (10, ui_y)); ui_y += 24
            screen.blit(font.render(f"Blue : {team_scores['Blue']}", True, BLUE),  (10, ui_y)); ui_y += 24
            screen.blit(font.render(f"Time : {remain_sec}s", True, WHITE), (10, ui_y)); ui_y += 24
            screen.blit(font.render(f"G HP {tank1.life} Fuel {int(tank1.fuel)} Ammo {tank1.ammo}", True, GREEN), (10, ui_y)); ui_y += 24
            screen.blit(font.render(f"B HP {tank2.life} Fuel {int(tank2.fuel)} Ammo {tank2.ammo}", True, BLUE),  (10, ui_y)); ui_y += 24
            screen.blit(font.render("ESC to quit (CSV auto saved)", True, WHITE), (10, SCREEN_HEIGHT - 26))

            if game_over:
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                overlay.fill((0,0,0,150))
                screen.blit(overlay, (0,0))
                wt = big_font.render(winner_text, True, WHITE)
                screen.blit(wt, wt.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)))

            pygame.display.flip()

    finally:
        logger.close()
        print(f"[INFO] CSV 已關閉並保存: {logger.path}")

if __name__ == "__main__":
    main()

