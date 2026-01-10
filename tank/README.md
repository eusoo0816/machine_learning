Tank battle-Decision tree
===
本系統目標為建構一套以監督式學習為基礎的遊戲 AI 決策系統，透過大量對戰資料蒐集，使用 Decision Tree（決策樹）分類器，學習坦克在不同戰場情境下的行為決策。  
本與傳統以單一規則或單一模型決策不同，本系統採用分層式（Hierarchical）決策架構，將複雜行為拆解為多個子任務，以提升穩定度與可解釋性。

1.需求
---
1-1 功能需求 (Functional Requirements)
<img width="1064" height="713" alt="image" src="https://github.com/user-attachments/assets/30643466-864c-4eaa-b9d6-887a39ec1d3d" />

可載入模型與ai對打

綠色方:策略樹模型
藍色方:AI


1-2 效能需求(Performance Requirements)
---
1.訓練後能使模型有明顯性能提升(如:開火頻率、命中率、閃躲能力等...)  
2.遊戲分數隨資料蒐集數量呈現上升趨勢  
<img width="620" height="471" alt="image" src="https://github.com/user-attachments/assets/ea32d999-e72d-4c06-aebf-948ae9660a0d" />  
陰影是資料分數分布，中間的藍線是平均值

1-3 介面需求(Interface Requirements)
---
遊戲介面
<img width="1064" height="713" alt="image" src="https://github.com/user-attachments/assets/3ac66d99-bfe3-414f-a08a-3be76a81a889" />



外部介面:可透過按鈕操控坦克前、後、左、右移動，開火按鍵可射擊  
內部介面:使用TANKTREEtrain產生joblib檔，遊戲平台程式python可讀取joblib檔

1-4 限制(Constraints)
---
語言:python  
環境版本:python3.13.7  
作業系統:Windows 11專業版64位元

硬體列表:  
處理器:Intel(R) Core(TM) i7-9700KF CPU @ 3.60GHz  
顯卡型號:NVIDIA GeForce GT 730

breakdown
---
<img width="1019" height="468" alt="image" src="https://github.com/user-attachments/assets/c185058a-d63e-4454-8961-8fe0553028e7" />  


設計
---
<img width="819" height="483" alt="image" src="https://github.com/user-attachments/assets/3b1f4371-4e90-4f78-bea1-af01ed64ea92" />  


---
API
---


| 項目 | 說明 |
|------|------|
| **`ModelTankAI.decide`** | |
| 功能 | Choose_action |
| 輸入 | `tank` (自身), `enemy` (敵人), `supplies` (補給), `walls` (牆), `bullets` (子彈) |
| 輸出 | `(move_cmd, fire_cmd, turret_cmd)` (移動指令, 開火指令, 砲塔角度) |
| 方法邏輯 | 1. 提取遊戲特徵 (`build_features`)<br>2. 呼叫 `goal_model` 預測大目標 (0:戰鬥, 1:彈藥, 2:油料, 3:閃避)<br>3. 根據目標呼叫對應的移動模型 (`move_fight` 或 `move_supply`)<br>4. 呼叫 `fire_model` 決定是否開火 |
| 使用方法 | 在遊戲主迴圈中每一幀呼叫，用來獲取 AI 下一步的行動 |

| 項目 | 說明 |
|------|------|
| **`JoblibPolicy.predict_one`** | |
| 功能 | Model_load (使用載入的模型進行預測) |
| 輸入 | `feat_dict` (當前遊戲狀態的特徵字典) |
| 輸出 | `int` (預測出的類別，如動作 ID) |
| 方法邏輯 | 將字典轉換為 Pandas DataFrame，並呼叫 `self.model.predict()` |
| 使用方法 | 被 `ModelTankAI.decide` 內部呼叫，用來取得具體的 Goal 或 Move 結果 |

| 項目 | 說明 |
|------|------|
| **`train_classifier`** | |
| 功能 | train, Model_save |
| 輸入 | `X` (特徵數據), `y` (標籤), `sample_weight` (權重), `model_name` (儲存名稱) |
| 輸出 | `clf` (訓練好的決策樹模型物件) |
| 方法邏輯 | 1. 切割訓練集與測試集 (`train_test_split`)<br>2. 建立並訓練決策樹 (`DecisionTreeClassifier.fit`)<br>3. 評估準確率並印出報告<br>4. 使用 `joblib.dump` 將模型存檔 (Model_save) |
| 使用方法 | 執行 `TANKTREEtrain.py` 的 main 函數時自動呼叫，用於生成 `.joblib` 模型檔 |

| 項目 | 說明 |
|------|------|
| **`Tank.update`** | |
| 功能 | Move tank |
| 輸入 | `walls` (僅在 Playtree 版本中需要傳入以檢測碰撞) |
| 輸出 | 無 (直接修改物件內部的 `x`, `y`, `fuel` 屬性) |
| 方法邏輯 | 1. 檢查 `move_forward`, `move_left` 等布林值<br>2. 更新座標 `x += dx`, `y += dy`<br>3. 扣除油料 (`fuel`)<br>4. 限制邊界 (`clamp`) 與處理牆壁碰撞 |
| 使用方法 | 在遊戲主迴圈中，每一幀對所有存活的坦克呼叫一次 |





決策樹(Decision tree)
---
https://github.com/eusoo0816/machine_learning/issues/5  

Loss Function  
---
<img width="459" height="331" alt="image" src="https://github.com/user-attachments/assets/f4e3521a-a26a-4bb6-8068-6bcdf46b5a98" />  

Learning Curve (Loss) - label=action  
決定『要做什麼動作』

Log Loss(縱軸):  
模型犯錯的嚴重程度

Training Samples:  
多少筆資料來訓練模型

Train Log Loss:模型在「看過的資料」上，判斷時「有沒有亂選」。  

Val Log Loss:模型在「沒看過的資料」上，判斷時「有沒有亂選」。  



資料蒐集策略
---
將遊戲內子彈數調高使
ai策略為一直開火  
1.會閃子彈  
2.會一直吃補包(油、彈藥)  
蒐集資料:https://github.com/eusoo0816/machine_learning/issues/2

成果
===
demo影片:https://www.youtube.com/watch?v=XGUjDp7KjfQ
