import KNN_main

# 建立實體
knn = KNN_main.PingPongKNN(model_path='my_game_model_NEW_2P.pkl', k=5)

# 訓練 (指向你的 LOG 資料夾)
# 假設你的 log 都在 ./logs 資料夾下
knn.train(data_folder='./trainning_NEW_2P')

# 儲存
knn.save()
