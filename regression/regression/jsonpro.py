# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import random

# JSONファイルからデータを読み込む関数
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return np.array(data['X']), np.array(data['Y'])

# モデルを作成する関数
def create_model(X, Y):
    a = random.randint(10, 60)
    b = random.randint(0, 2000)
    value = np.mean((Y - (a * X + b)) ** 2)
    
    for _ in range(10000):
        c = random.randint(-5, 5)
        d = random.randint(-50, 50)
        new_value = np.mean((Y - ((a-c) * X + (b-d))) ** 2)
        if new_value < value:
            value = new_value
            a, b = a-c, b-d
    
    return a, b

# データの読み込み
X_train, Y_train = load_data('train_data.json')
X_test, Y_test = load_data('test_data.json')

# モデルの作成
a, b = create_model(X_train, Y_train)
print(f"作成されたモデル: Y = {a}X + {b}")

# テストデータに対する予測
Y_pred = a * X_test + b

# 平均二乗誤差（MSE）の計算
mse = np.mean((Y_test - Y_pred) ** 2)
print(f"平均二乗誤差 (MSE): {mse:.2f}")

# 決定係数（R^2）の計算
ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)
ss_res = np.sum((Y_test - Y_pred) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"決定係数 (R^2): {r_squared:.4f}")

# 結果のプロットと残差プロットを同時に表示
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# 結果のプロット
ax1.scatter(X_test, Y_test, color='blue', label='実際のデータ')
ax1.plot(X_test, Y_pred, color='red', label='予測')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('テストデータに対する予測結果')
ax1.legend()
ax1.grid(True)

# 残差プロット
residuals = Y_test - Y_pred
ax2.scatter(X_test, residuals)
ax2.axhline(y=0, color='r', linestyle='-')
ax2.set_xlabel('X')
ax2.set_ylabel('残差')
ax2.set_title('残差プロット')
ax2.grid(True)

# グラフ全体のタイトル
fig.suptitle('モデル評価結果', fontsize=16)

# グラフ間の間隔を調整
plt.tight_layout()

# グラフを表示
plt.show()