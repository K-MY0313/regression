import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colors
import matplotlib.animation
from IPython.display import HTML
import japanize_matplotlib
import random
import sys
import io
import random
# 標準出力のエンコーディングを設定
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# JSONファイルからデータを読み込む
with open('ice.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

X = np.array(data['X'])
Y = np.array(data['Y'])
D = np.column_stack((X, Y))  # 2次元配列に変換
value = 0  # 初期値を設定
squaredresiduals = 0
i = 0
count = 0
learningrate = random.uniform(1e-20, 1e-30)
a = random.randint(20, 50)
b = 0
#標準化
X_mean = np.mean(X)
X_std = np.std(X)
X_normalized = (X - X_mean) / X_std
Y_mean = np.mean(Y)
Y_std = np.std(Y)
Y_normalized = (Y - Y_mean) / Y_std
for _ in iter(int, 1):
 for i in range(1000):
    #求める2次関数
    prediction = a * X_normalized + b
    #求めた残差
    error = prediction - Y_normalized
    #右辺でa(傾き)の勾配(変化率)を求め、新たな傾きを求める
    a -= learningrate * np.dot(error, X_normalized) / len(X_normalized)
    #左辺でb(切片)の勾配(変化率)を求め、新たな切片を求める
    b -= learningrate * np.sum(error) / len(X_normalized)
    #合計残差が0.1以下になればOK
    if np.mean(error**2) < 15000:
        break
 learningrate *= 10
 if np.mean(error**2) < 20000:
        print(np.mean(error**2))
        break
# モデルの評価結果を表示
print(f"このモデルは{a}x+{b}です。mseは{np.mean(error**2)}".encode('utf-8').decode('utf-8'))
X_original = X_normalized * X_std + X_mean
prediction_original = (prediction * Y_std + Y_mean) if Y_normalized is not None else prediction
# グラフの描画
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='データポイント')
plt.plot(X, a * X + b, color='red', label=f'フィットしたモデル: {a}x + {b}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('データポイントとフィットしたモデル')
plt.legend()
plt.grid(True)
plt.show()
