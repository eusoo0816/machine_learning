# ml_play_pong_god_v3.py
# PingPong「地獄模式 v3：blocker + 戰術 + 切球」AI
#
# 特點：
#   1. 用完整模擬（左右牆 + 上下牆 + 對手板子線 + blocker）預測
#      「下一次球回到自己板子那條 y 線」時的 x。
#   2. 回球時會把站位往「離對手板子更遠的一側」偏移一些，讓球飛向對手難接的位置。
#   3. 球快撞上板子且已經對準時，利用切球機制：
#      - 若想讓球往某一側飛，就選擇 MOVE_LEFT / MOVE_RIGHT 去控制 vx 的方向。

import random
import os
import time
import json

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.side = ai_name
        self.ball_served = False
        self.log = []
        game_params = kwargs.get("game_params", {})
        print(f"[{self.side}] HELL MODE v3 params:", game_params)

        # 直接使用題目規格（不要再信 game_params 亂七八糟）
        self.WIDTH = 200
        self.HEIGHT = 500
        self.BALL_SIZE = 10
        self.PLATFORM_WIDTH = 40
        self.PLATFORM_HEIGHT = 10
        self.BLOCKER_WIDTH = 30
        self.BLOCKER_HEIGHT = 20

        # 固定 y 位置（top-left）
        self.P1_Y = 420
        self.P2_Y = 70
        self.BLOCKER_Y = 240

        # blocker 速度估計用
        self.prev_blocker = None
        self.blocker_vx = 0.0

    # ---------------- 工具：模擬直到下一次打到自己板子線 ---------------- #
    def _predict_next_hit_x(self,
                            ball_x, ball_y, vx, vy,
                            my_y, opp_y,
                            blocker_pos, blocker_vx):
        """
        使用離散模擬（以 top-left 座標）預測：
        「下一次球通過 my_y（自己板子線）」時的 x。

        模擬時考慮：
        - 左右牆反彈（以 BALL_SIZE 當寬度）
        - 上下牆反彈
        - 對手板子線 opp_y：視為「無限寬板子，必接」，vy 反向
        - HARD 模式 blocker：
            * blocker 以 blocker_vx 水平移動，撞到左右邊會反彈
            * 球矩形與 blocker 矩形相交時，簡化為反轉 vy（往上或往下）
        """

        # ball top-left
        x = float(ball_x)
        y = float(ball_y)
        vx = float(vx)
        vy = float(vy)

        # blocker 初始
        if blocker_pos is not None:
            blk_x, blk_y = blocker_pos
            blk_x = float(blk_x)
            blk_y = float(blk_y)
            blk_vx = float(blocker_vx)
            blk_max_x = self.WIDTH - self.BLOCKER_WIDTH
        else:
            blk_x = blk_y = blk_vx = blk_max_x = None

        max_ball_x = self.WIDTH - self.BALL_SIZE
        max_ball_y = self.HEIGHT - self.BALL_SIZE

        max_steps = 5000

        for _ in range(max_steps):
            prev_x, prev_y = x, y

            # 1) 球先走一步
            x += vx
            y += vy

            # 2) 左右牆反彈
            if x < 0:
                x = -x
                vx = -vx
            elif x > max_ball_x:
                x = 2 * max_ball_x - x
                vx = -vx

            # 3) 上下牆反彈
            if y < 0:
                y = -y
                vy = -vy
            elif y > max_ball_y:
                y = 2 * max_ball_y - y
                vy = -vy

            # 4) 更新 blocker + 染進模擬 ••••••••••••••••••
            if blk_x is not None:
                blk_x += blk_vx
                if blk_x < 0:
                    blk_x = -blk_x
                    blk_vx = -blk_vx
                elif blk_x > blk_max_x:
                    blk_x = 2 * blk_max_x - blk_x
                    blk_vx = -blk_vx

                # 球 vs blocker 矩形相交檢測
                ball_left = x
                ball_right = x + self.BALL_SIZE
                ball_top = y
                ball_bottom = y + self.BALL_SIZE

                blk_left = blk_x
                blk_right = blk_x + self.BLOCKER_WIDTH
                blk_top = blk_y
                blk_bottom = blk_y + self.BLOCKER_HEIGHT

                if (ball_right >= blk_left and ball_left <= blk_right and
                        ball_bottom >= blk_top and ball_top <= blk_bottom):
                    # 簡化：只處理垂直反彈
                    if vy > 0:
                        y = blk_top - self.BALL_SIZE
                    else:
                        y = blk_bottom
                    vy = -vy

            # 5) 先檢查是否穿過「對手板子線」 → 必接，vy 反向
            if (prev_y - opp_y) * (y - opp_y) <= 0:
                if y != prev_y:
                    alpha = (opp_y - prev_y) / (y - prev_y)
                    hit_x = prev_x + alpha * (x - prev_x)
                else:
                    hit_x = x

                x = hit_x
                y = opp_y
                vy = -vy

            # 6) 再檢查是否穿過「自己板子線」 → 這就是我們要的落點
            if (prev_y - my_y) * (y - my_y) <= 0:
                if y != prev_y:
                    alpha = (my_y - prev_y) / (y - prev_y)
                    land_x = prev_x + alpha * (x - prev_x)
                else:
                    land_x = x
                return land_x

        # 模擬太久還沒打到自己 → 退而求其次回傳當前 x
        return x

    # ---------------- 工具：板子移動 ---------------- #
    def _move_to_center(self, plat_x, target_center, dead_zone=1.0):
        my_center = plat_x + self.PLATFORM_WIDTH / 2
        if my_center < target_center - dead_zone:
            return "MOVE_RIGHT"
        elif my_center > target_center + dead_zone:
            return "MOVE_LEFT"
        else:
            return "NONE"

    # ---------------- 主邏輯 ---------------- #
    def update(self, scene_info, *args, **kwargs):
        status = scene_info["status"]

        # 遊戲結束
        if status != "GAME_ALIVE":
            self.ball_served = False
            self.prev_blocker = None
            self.blocker_vx = 0.0

            log_folder = "C:/Users/gslab/Desktop/pingpong1/trainning_NEW"
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
                
            log_file_path = os.path.join(log_folder, f"{time.time()}.json")

            try:
                with open(log_file_path, "r", encoding='utf-8') as log_file:
                    log_data = json.load(log_file)
            except FileNotFoundError:
                log_data = []

            log_data = self.log

            with open(log_file_path, "w", encoding='utf-8') as log_file:
                json.dump(log_data, log_file, ensure_ascii=False, indent=4) 


            return "RESET"

        # 讀場景資訊
        ball_x, ball_y = scene_info["ball"]
        ball_vx, ball_vy = scene_info["ball_speed"]
        plat1_x, plat1_y = scene_info["platform_1P"]
        plat2_x, plat2_y = scene_info["platform_2P"]
        blocker_pos = scene_info.get("blocker", None)

        # 更新 blocker 速度估計
        if blocker_pos is not None:
            if self.prev_blocker is not None:
                prev_bx, _ = self.prev_blocker
                bx, _ = blocker_pos
                self.blocker_vx = bx - prev_bx
            self.prev_blocker = blocker_pos
        else:
            self.prev_blocker = None
            self.blocker_vx = 0.0

        # 分辨自己是哪一邊
        if self.side == "1P":
            my_plat_x, my_plat_y = plat1_x, plat1_y
            opp_plat_x, opp_plat_y = plat2_x, plat2_y
            my_y_line = self.P1_Y
            toward_me = (ball_vy > 0)
        else:
            my_plat_x, my_plat_y = plat2_x, plat2_y
            opp_plat_x, opp_plat_y = plat1_x, plat1_y
            my_y_line = self.P2_Y
            toward_me = (ball_vy < 0)

        my_center = my_plat_x + self.PLATFORM_WIDTH / 2
        opp_center = opp_plat_x + self.PLATFORM_WIDTH / 2
        ball_center = ball_x + self.BALL_SIZE / 2
        mid_x = self.WIDTH / 2

        ball_served_flag = scene_info["ball_served"]

        # ---------------- 1) 發球策略：打到離對手更遠那邊 ---------------- #
        if not ball_served_flag and not self.ball_served:
            self.ball_served = True
            if opp_center < mid_x:
                return "SERVE_TO_RIGHT"
            elif opp_center > mid_x:
                return "SERVE_TO_LEFT"
            else:
                return random.choice(["SERVE_TO_LEFT", "SERVE_TO_RIGHT"])

        # ---------------- 2) 模擬下一次回到自己板子線的落點 ---------------- #
        landing_x = self._predict_next_hit_x(
            ball_x, ball_y, ball_vx, ball_vy,
            my_plat_y, opp_plat_y,
            blocker_pos,
            self.blocker_vx
        )

        self.log.append({"ball": scene_info['ball'], "ball_speed": scene_info['ball_speed'], "frame": scene_info['frame'], "landing_x": landing_x})
        
        # 模擬得到的是 ball 的「左上角」x，轉成中心
        base_target_center = landing_x + self.BALL_SIZE / 2

        # ---------------- 3) 戰術：站位偏移，打遠離對手的那一側 ---------------- #
        # 想把球打往「離對手更遠」的方向
        if opp_center < mid_x:
            # 對手偏左 → 我們想把球打去右側
            desired_dir = +1
            offset_sign = +1
        elif opp_center > mid_x:
            # 對手偏右 → 我們想把球打去左側
            desired_dir = -1
            offset_sign = -1
        else:
            desired_dir = 0
            offset_sign = random.choice([-1, +1])

        # 根據球與自己的垂直距離，調整偏移大小
        dist_y = abs(ball_y - my_plat_y)
        half_w = self.PLATFORM_WIDTH / 2
        max_offset = half_w * 0.6  # 最多偏 60% 的半寬，保守一點

        if dist_y > 200:
            offset_factor = 1.0   # 還很遠，可以偏多一些
        elif dist_y > 100:
            offset_factor = 0.6
        elif dist_y > 60:
            offset_factor = 0.3
        else:
            offset_factor = 0.0   # 很近了，偏移會增加 miss 風險

        tactical_offset = offset_sign * max_offset * offset_factor
        target_center = base_target_center + tactical_offset

        # 邊界限制
        min_center = half_w
        max_center = self.WIDTH - half_w
        if target_center < min_center:
            target_center = min_center
        if target_center > max_center:
            target_center = max_center

        # ---------------- 4) 球非常靠近 → 務必要對準球中心 ---------------- #
        # 防止預測誤差：一旦球很接近自己板子，就優先鎖球中心
        if dist_y < 50:
            target_center = ball_center

        # ---------------- 5) 切球：在「很近且已對準」時控制 vx 方向 ---------------- #
        #   利用題目的切球規則：
        #     - 板子與球的 X 方向相同 → |vx| += 3（方向不變）
        #     - 板子與球的 X 方向相反 → vx 反向
        #     - 板子不動 → vx 不變
        #
        #   我們希望：
        #     - 若對手在左邊 → 球往右飛（vx > 0）
        #     - 若對手在右邊 → 球往左飛（vx < 0）
        #
        #   當球很近、而且我們已經對準球中心時，使用 MOVE_LEFT / MOVE_RIGHT 來達成。
        align_tol = 2.0
        if dist_y < 45 and abs(my_center - ball_center) < align_tol and desired_dir != 0:
            # 這裡假設「下一幀內可能會撞到球」，所以直接選方向
            def sign(v):
                return 0 if v == 0 else (1 if v > 0 else -1)

            cur_sign = sign(ball_vx)

            if cur_sign == 0:
                # vx = 0 → 我們用移動方向產生想要的方向
                if desired_dir > 0:
                    return "MOVE_RIGHT"
                else:
                    return "MOVE_LEFT"
            else:
                # vx != 0 → 決定要保持或翻轉
                if cur_sign == desired_dir:
                    # 想要保持方向 → 板子跟球同向移動，讓 |vx| 增加
                    if cur_sign > 0:
                        return "MOVE_RIGHT"
                    else:
                        return "MOVE_LEFT"
                else:
                    # 想要翻轉方向 → 板子與球反向移動
                    if cur_sign > 0:
                        return "MOVE_LEFT"
                    else:
                        return "MOVE_RIGHT"

        # ---------------- 6) 一般情況：往 target_center 移動 ---------------- #
        return self._move_to_center(my_plat_x, target_center, dead_zone=1.0)

    def reset(self):
        print(f"reset {self.side}")
        self.ball_served = False
        self.prev_blocker = None
        self.blocker_vx = 0.0
