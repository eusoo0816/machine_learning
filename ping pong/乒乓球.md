## PingPong AI 專案 API 文件
---

### 需求

- 功能:
  - 球拍能夠左右移動
  - 發球
  - 預測球落點位置
  - 預測對方回擊路線
- 系統:
  - Window10、11
  - Python3.11
  - numpy 1.26.4
---

## 介面

- 限制
  - 螢幕大小 200 x 500
  - 板子 40 x 10
  - 球 10 x 10
  - 障礙物 30 x 20
---
## Breakdown

<img width="2067" height="984" alt="image" src="https://github.com/user-attachments/assets/91daeddb-e267-4ff5-81b7-2941a8c4b1be" />

---
## 設計

<img width="935" height="393" alt="image" src="https://github.com/user-attachments/assets/81db817d-b036-4a9d-b17c-9b58b4d63290" />

---
## Model-KNN分類

<img width="1376" height="459" alt="image" src="https://github.com/user-attachments/assets/91b2371a-ca75-43ea-a15b-97599453d2e8" />

---
## Model-KNN回歸

<img width="710" height="228" alt="image" src="https://github.com/user-attachments/assets/5f2134c1-63dc-4378-aa74-2ef13c48dcfe" />

---
## 驗收
 - 根據需求進行單元測試，完成需求驗收
 - 驗收條件是從功能角度出發，確保每項功能都能夠符合設計需求，並提供準確、穩定的操作。

**1. 球拍能夠左右移動**
- 功能描述：
  - 球拍能夠在水平軸上進行左右平移，並能夠準確地跟隨球的運動軌跡進行調整，保持對球的控制。
- 驗收條件：
  - 平移範圍：球拍必須能夠在預定的範圍內平移，並且在此範圍內運行平穩。
  - 反應速度：球拍的左右平移應具備即時反應能力，對發球或回擊過來的球能夠迅速做出調整，且無顯著延遲。
  - 精確度：球拍的平移過程應保持高精度，確保其能夠準確地與球的接觸點對齊。
  - 穩定性：無論在高速度或突發情況下，球拍的平移功能應穩定運作，不會有跳動、錯位或卡頓現象。
- 驗收成果：可以精準且每幀都能做到移動
    
**2. 發球**
- 功能描述：
  - 可以進行發球操作，並且發球能夠依照設置的規範與預期進行。
- 驗收條件：
  - 發球準確性：發出的球應符合設置的發球方向、時間。
  - 發球穩定性：無論操作次數多少，發球機制應無故障。
- 驗收成果：可正常發球
    
**3. 預測球落點位置**
- 功能描述：
  - 系統能夠準確預測來球在擊中地面後的落點位置，並根據此落點指導球拍操作。
- 驗收條件：
  - 預測精度：系統能夠根據球的速度、角度、彈性等因素準確預測反彈軌跡，誤差範圍應在可接受的範圍內。
  - 反應時間：系統應在極短時間內計算並預測的落點位置。
  - 穩定性：無論球速多快，預測結果應始終穩定且一致
- 驗收成果：可準確預判落點位置
      
**4. 預測對方回擊路線**
- 功能描述：
  -  系統能夠預測對手回擊的球路，包括球的速度、方向和角度，並根據此信息調整球拍位置。
- 驗收條件：
  - 準確度：AI能夠準確預測對手回擊的方向、角度與速度，誤差範圍應小於可接受的限度（例如誤差範圍小於5或1）。
  - 即時性：預測必須在對手回擊動作完成後的瞬間即時更新，無延遲現象。
  - 預測穩定性：系統預測的回擊路線應穩定可靠，不受外部因素如環境變化或隨機性影響(障礙物)。
- 驗收成果：可先預測對手可能的回擊路線，並先移動到落點位置
  
---
## MLGame AI 介面：`class MLPlay`

**用途**

- 對應 MLGame 的標準介面：框架每一幀傳入 `scene_info`，AI 回傳一個指令字串。
- 會在遊戲結束時把每幀紀錄（含 `landing_x`）寫成 JSON 檔（用於訓練資料蒐集）。

### `MLPlay.__init__(ai_name, *args, **kwargs)`

| 欄位 | 說明 |
|---|---|
| 函數名稱 | `__init__` |
| 輸入 | `ai_name: str`（通常 `"1P"` / `"2P"`）<br>`kwargs["game_params"]: dict`（可選，用於印出參數） |
| 輸出 | `None` |
| 參數/狀態 | `self.side`：AI 所在方（`"1P"` / `"2P"`）<br>`self.ball_served`：是否已發球（`bool`）<br>`self.log`：每幀紀錄（`list[dict]`）<br>`self.prev_blocker / self.blocker_vx`：blocker 估速<br>場地常數：`WIDTH=200`、`HEIGHT=500`、`BALL_SIZE=10`、`PLATFORM_WIDTH=40`、`BLOCKER_WIDTH=30`…<br>（KNN 版本）`self.knn = PingPongKNN(model_path='my_game_model.pkl')` 並呼叫 `load()` |
| 方法 | 建立 AI 內部狀態；KNN 版本會嘗試載入模型檔 `my_game_model.pkl`（若不存在會印錯誤，預測時可能回傳 `None`）。 |
| 使用方法 | 由遊戲框架載入並建立物件（通常不需要手動呼叫）。 |

### `MLPlay.update(scene_info, *args, **kwargs) -> str`

| 欄位 | 說明 |
|---|---|
| 函數名稱 | `update` |
| 輸入 | `scene_info: dict`（依程式使用的常見鍵）<br>- `status: str`（如 `"GAME_ALIVE"`）<br>- `ball: (x, y)`（球左上角）<br>- `ball_speed: (vx, vy)`<br>- `platform_1P: (x, y)`（板子左上角）<br>- `platform_2P: (x, y)`<br>- `blocker: (x, y)`（可選）<br>- `ball_served: bool`<br>- `frame: int` |
| 輸出 | `str` 指令（見下方「回傳指令」） |
| 參數/狀態 | 依 `self.side` 判斷 1P/2P；更新 `self.ball_served`、blocker 估速狀態、以及 `self.log`。 |
| 方法 | **遊戲結束**（`status != "GAME_ALIVE"`）：重設狀態 → 將 `self.log` 寫入 JSON → 回傳 `"RESET"`。<br>**尚未發球**：依對手板子站位回傳 `"SERVE_TO_LEFT"` 或 `"SERVE_TO_RIGHT"`。<br>**一般回合**：計算 `landing_x`（模擬版呼叫 `_predict_next_hit_x`；KNN 版呼叫 `self.knn.predict(...)`）→ append 到 `self.log` → 回傳 `"MOVE_LEFT"` / `"MOVE_RIGHT"` / `"NONE"`。 |
| 使用方法 | 由遊戲框架每幀呼叫：`command = ai.update(scene_info)` |

### `MLPlay.reset()`

| 欄位 | 說明 |
|---|---|
| 函數名稱 | `reset` |
| 輸入 | 無 |
| 輸出 | `None` |
| 參數/狀態 | 重設 `self.ball_served`、`self.prev_blocker`、`self.blocker_vx` |
| 方法 | 重設狀態並印出 `reset {self.side}` |
| 使用方法 | 由遊戲框架在重置時呼叫（通常不需要手動呼叫）。 |

### （內部輔助）`MLPlay._predict_next_hit_x(...) -> float`

> 以底線開頭，屬於內部工具函數；但若你要改 AI 或做單元測試，這是最核心的落點預測工具之一。

| 欄位 | 說明 |
|---|---|
| 函數名稱 | `_predict_next_hit_x` |
| 輸入 | `ball_x, ball_y: float|int`（球左上角）<br>`vx, vy: float|int`（球速度）<br>`my_y: float|int`（自己板子 y 線）<br>`opp_y: float|int`（對手板子 y 線）<br>`blocker_pos: (x, y) | None`<br>`blocker_vx: float|int` |
| 輸出 | `land_x: float`（球下一次穿過 `my_y` 時的 x；以球左上角座標系） |
| 參數/狀態 | 使用 `self.WIDTH / self.HEIGHT / self.BALL_SIZE / self.BLOCKER_*` 等常數做模擬。 |
| 方法 | 離散步進模擬球運動：牆反彈 + 對手線必接（vy 反向）+ blocker 碰撞（簡化 vy 反向）。若步數超過上限仍未回到 `my_y`，回傳當前 `x` 作為退而求其次。 |
| 使用方法 | 通常由 `update()` 內部呼叫（模擬版 AI）。 |

### （內部輔助）`MLPlay._move_to_center(plat_x, target_center, dead_zone=1.0) -> str`

| 欄位 | 說明 |
|---|---|
| 函數名稱 | `_move_to_center` |
| 輸入 | `plat_x: float|int`（板子左上角 x）<br>`target_center: float|int`（目標中心 x）<br>`dead_zone: float = 1.0`（死區） |
| 輸出 | `"MOVE_LEFT"` / `"MOVE_RIGHT"` / `"NONE"` |
| 參數/狀態 | 使用 `self.PLATFORM_WIDTH` 計算板子中心位置。 |
| 方法 | 比較板子中心與 `target_center`，決定移動方向（含死區）。 |
| 使用方法 | 通常由 `update()` 內部呼叫。 |

### 回傳指令（`update()`）

- **發球**：`"SERVE_TO_LEFT"`、`"SERVE_TO_RIGHT"`
- **移動**：`"MOVE_LEFT"`、`"MOVE_RIGHT"`、`"NONE"`
- **重置**：`"RESET"`

### 訓練資料輸出（JSON）

`update()` 在遊戲結束時會把 `self.log` 寫入 JSON；每筆資料格式如下：

```json
{
  "ball": [ball_x, ball_y],
  "ball_speed": [vx, vy],
  "frame": frame_id,
  "landing_x": landing_x
}
```

**各檔案的 log 輸出資料夾**

- `AI_1P.py`：`trainning_1P/`
- `AI_2P.py`：`trainning_2P/`
  
---

## KNN 模型工具：`class PingPongKNN`

**出現位置（去重後建議）**

- **建議使用**：`KNN/KNN_main.py`
- **亦存在同名複本**：`AI_KNN_1P.py`、`AI_KNN_2P.py`（內容等價，避免重複說明）

**用途**

- 使用 KNN（`KNeighborsRegressor`）從特徵 \([ball_x, ball_y, speed_x, speed_y]\) 回歸預測 `landing_x`。
- 內建 `MinMaxScaler` 做特徵標準化，並把 `(model, scaler)` 一起存檔。

### `PingPongKNN.train(data_folder, file_pattern='*.json') -> bool`

| 欄位 | 說明 |
|---|---|
| 函數名稱 | `train` |
| 輸入 | `data_folder: str`（JSON 訓練資料資料夾）<br>`file_pattern: str = '*.json'` |
| 輸出 | `True`：成功讀到資料並完成訓練<br>`False`：找不到檔案或沒有有效訓練資料 |
| 參數/狀態 | 會建立/更新 `self.scaler`、`self.model`，並將 `self.is_loaded=True` |
| 方法 | 讀取 JSON → 抽取 `features=[ball_x, ball_y, speed_x, speed_y]` 與 `label=landing_x` → `MinMaxScaler` 標準化 → 訓練 `KNeighborsRegressor(n_neighbors=k, weights='distance')`。 |
| 使用方法 | `ok = knn.train(data_folder="trainning_NEW", file_pattern="*.json")` |

### `PingPongKNN.save()`

| 欄位 | 說明 |
|---|---|
| 函數名稱 | `save` |
| 輸入 | 無 |
| 輸出 | `None` |
| 參數/狀態 | 依 `self.model_path` 寫入 pickle 檔 |
| 方法 | 將 `(self.model, self.scaler)` 以 pickle 寫入 `self.model_path`。若 `self.is_loaded == False` 會印錯誤並返回。 |
| 使用方法 | `knn.save()` |

### `PingPongKNN.load() -> bool`

| 欄位 | 說明 |
|---|---|
| 函數名稱 | `load` |
| 輸入 | 無 |
| 輸出 | `True`：載入成功<br>`False`：找不到檔案或載入失敗 |
| 參數/狀態 | 載入成功會更新 `self.model`、`self.scaler`，並設 `self.is_loaded=True` |
| 方法 | 從 `self.model_path` 讀取 pickle，還原 `self.model` 與 `self.scaler`。 |
| 使用方法 | `ok = knn.load()` |

### `PingPongKNN.predict(ball_x, ball_y, speed_x, speed_y) -> float | None`

| 欄位 | 說明 |
|---|---|
| 函數名稱 | `predict` |
| 輸入 | `ball_x, ball_y, speed_x, speed_y: float|int` |
| 輸出 | `landing_x: float`<br>若模型未載入且 `load()` 失敗：回傳 `None` |
| 參數/狀態 | 需要 `self.is_loaded == True`（若為 False 會先自動嘗試 `load()`） |
| 方法 | 使用訓練時的 `scaler` 對輸入標準化後，呼叫 `self.model.predict()`，回傳第一個預測值。 |
| 使用方法 | `landing_x = knn.predict(ball_x, ball_y, vx, vy)` |

---

## 使用方法（最常見）

### 1) 用 `PingPongKNN` 訓練並存模型

```python
from KNN.KNN_main import PingPongKNN

knn = PingPongKNN(model_path="my_game_model.pkl", k=3)
ok = knn.train(data_folder="trainning_NEW", file_pattern="*.json")
if ok:
    knn.save()
```

### 2) 在 AI（`MLPlay`）中使用 KNN 預測落點

> `AI_KNN_1P.py` / `AI_KNN_2P.py` 已內建此流程：在 `__init__` 載入模型、在 `update()` 呼叫 `predict()`。

```python
from KNN.KNN_main import PingPongKNN

knn = PingPongKNN(model_path="my_game_model.pkl")
knn.load()
landing_x = knn.predict(ball_x, ball_y, vx, vy)
```
分工表
===
<img width="790" height="89" alt="image" src="https://github.com/user-attachments/assets/1ead9237-d550-4298-82ba-5e85e2f5b8a8" />
