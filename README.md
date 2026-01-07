Tank battle-Decision tree
===
本系統目標為建構一套以監督式學習為基礎的遊戲 AI 決策系統，透過大量對戰資料蒐集，使用 Decision Tree（決策樹）分類器，學習坦克在不同戰場情境下的行為決策。
與傳統以單一規則或單一模型決策不同，本系統採用分層式（Hierarchical）決策架構，將複雜行為拆解為多個子任務，以提升穩定度與可解釋性。

1.需求
---
1-1 功能需求 (Functional Requirements)
---
1.遊戲主程式可載入模型進行遊玩
2.系統能引入遊戲數據(坦克位置、補包位置等...)作為輸入
3.核心功能:
  學習訓練
  策略評估
4.支援模式:
  蒐集模式:蒐集數據以供訓練。
  訓練模式:仔入蒐集數據供決策數進行訓練。
  遊玩模式:遊戲主程式能引入模型進行對打。
---
1-2 效能需求(Performance Requirements)
---
1.訓練後能使模型有明顯性能提升(如:開火頻率、命中率、閃躲能力等...)
2.遊戲分數隨資料蒐集數量呈現上升趨勢
---
1-3 介面需求(Interface Requirements)
---
breakdown
---
<img width="957" height="402" alt="image" src="https://github.com/user-attachments/assets/2cdcc1de-159f-4b8a-994c-43e5af0bc00b" />
---
架構圖
---

---
API
---
| 檔案名稱 | 1. 輸入 (Input) | 2. 輸出 (Output) | 3. 主要參數 (Param) | 4. 方法/邏輯 (Method) | 5. 呼叫例子 (Call Example) |
---
1-4 限制(Constraints)
---
1-5 驗收標準(Acceptance Criteria)
---
1.遊戲能載入模型與訓練用AI對打
2.能夠閃躲子彈
3.可以吃油補包
4.會朝敵人開火
5.會透過吃彈藥包補充子彈

---
決策樹(Decision tree)
---
<img width="873" height="596" alt="image" src="https://github.com/user-attachments/assets/7ae6b24b-cf2f-4c02-a010-0fcbe81ec50e" />

資料蒐集策略
---
將遊戲內子彈數調高
ai策略為一直開火
1.會閃子彈
2.會一直吃補包(油、彈藥)
資料參數:https://github.com/eusoo0816/machine_learning/issues/2

成果
===
demo影片:https://www.youtube.com/watch?v=XGUjDp7KjfQ
