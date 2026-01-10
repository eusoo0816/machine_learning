import os
import json
import glob
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

class PingPongKNN:
    def __init__(self, model_path='knn_model.pkl', k=3):
        """
        初始化 KNN 模型管理器
        :param model_path: 模型儲存與載入的路徑
        :param k: KNN 的鄰居數量 (預設 3)
        """
        self.model_path = model_path
        self.k = k
        self.model = None
        self.scaler = None
        self.is_loaded = False

    def train(self, data_folder, file_pattern='*.json'):
        """
        讀取資料夾內所有 LOG 並訓練模型
        """
        # 1. 搜尋所有 JSON 檔案
        search_path = os.path.join(data_folder, file_pattern)
        files = glob.glob(search_path)
        
        if not files:
            print(f"[Error] 在 {data_folder} 找不到任何 {file_pattern} 檔案")
            return False

        print(f"[Train] 找到 {len(files)} 個紀錄檔，開始讀取...")

        X_data = [] # 特徵
        y_data = [] # 答案 (落點)

        # 2. 讀取並解析資料
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data_list = json.load(f)
                    
                    for row in data_list:
                        # 檢查資料完整性
                        if 'ball' not in row or 'ball_speed' not in row or 'landing_x' not in row:
                            continue
                        
                        # 特徵: [Ball X, Ball Y, Speed X, Speed Y]
                        features = [
                            row['ball'][0],
                            row['ball'][1],
                            row['ball_speed'][0],
                            row['ball_speed'][1]
                        ]
                        label = row['landing_x']

                        X_data.append(features)
                        y_data.append(label)
            except Exception as e:
                print(f"[Warning] 讀取 {file_path} 失敗: {e}")

        if not X_data:
            print("[Error] 有檔案但沒有有效的訓練資料")
            return False

        # 轉為 Numpy Array
        X = np.array(X_data)
        y = np.array(y_data)

        print(f"[Train] 載入完成，共有 {len(X)} 筆資料。開始訓練...")

        # 3. 資料標準化 (非常重要)
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 4. 訓練 KNN
        self.model = KNeighborsRegressor(n_neighbors=self.k, weights='distance')
        self.model.fit(X_scaled, y)
        self.is_loaded = True
        
        print("[Train] 模型訓練完成！")
        return True

    def save(self):
        """
        將訓練好的模型與 Scaler 存檔
        """
        if not self.is_loaded:
            print("[Error] 模型尚未訓練，無法儲存")
            return

        try:
            with open(self.model_path, 'wb') as f:
                # 同時儲存模型與標準化工具
                pickle.dump((self.model, self.scaler), f)
            print(f"[Save] 模型已儲存至: {self.model_path}")
        except Exception as e:
            print(f"[Error] 儲存失敗: {e}")

    def load(self):
        """
        從檔案載入模型
        """
        if not os.path.exists(self.model_path):
            print(f"[Error] 找不到模型檔案: {self.model_path}")
            return False

        try:
            with open(self.model_path, 'rb') as f:
                self.model, self.scaler = pickle.load(f)
            self.is_loaded = True
            print(f"[Load] 模型載入成功: {self.model_path}")
            return True
        except Exception as e:
            print(f"[Error] 載入失敗: {e}")
            return False

    def predict(self, ball_x, ball_y, speed_x, speed_y):
        """
        輸入當前狀態，預測落點 X
        """
        if not self.is_loaded:
            # 如果忘記載入，嘗試自動載入一次
            if not self.load():
                return None

        # 1. 整理輸入
        input_data = np.array([[ball_x, ball_y, speed_x, speed_y]])

        # 2. 標準化 (必須使用跟訓練時一樣的 scaler)
        input_scaled = self.scaler.transform(input_data)

        # 3. 預測
        result = self.model.predict(input_scaled)
        
        return float(result[0])
