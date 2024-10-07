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
while True:  # 無限ループ
    a = random.randint(10, 60)
    b = random.randint(0, 2000)
    squaredresiduals = np.mean((Y - (a * X + b)) * (Y - (a * X + b)))
    print(squaredresiduals)
    if squaredresiduals <= 20000:
        # 値が20000以下になったらループを終了
        value = squaredresiduals 
        break
for i in range(10000):
    # 閾値c,d
    c = random.randint(-5, 5)
    d = random.randint(-50, 50)
   
    squaredresiduals = np.mean((Y - ((a-c) * X + (b-d))) * (Y - ((a-c) * X + (b-d))))
    if value > squaredresiduals:
        value = squaredresiduals
        a=a-c
        b=b-d
        count = 0
    if count == 100:
        break
    count = count + 1
# モデルの評価結果を表示
print(f"このモデルは{a}x+{b}です".encode('utf-8').decode('utf-8'))

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
