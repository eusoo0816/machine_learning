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
蒐集資料:https://github.com/eusoo0816/machine_learning/issues/4  
訓練模型:https://github.com/eusoo0816/machine_learning/issues/3





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
