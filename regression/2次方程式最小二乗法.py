import json
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# JSONデータの読み込み
with open('ice.json', 'r') as f:
    data = json.load(f)

# データの抽出と整形
x_data = np.array(data['X'])
y_data = np.array(data['Y'])

# シンボルの定義と方程式の構築
a, b, c = sp.symbols('a b c')
eq1 = sp.Eq(sum(x_data**2 * y_data), sum(x_data**4 * a) + sum(x_data**3 * b) + sum(x_data**2 * c))
eq2 = sp.Eq(sum(x_data * y_data), sum(x_data**3 * a) + sum(x_data**2 * b) + sum(x_data * c))
eq3 = sp.Eq(sum(y_data), sum(x_data**2 * a) + sum(x_data * b) + len(x_data))

# 連立方程式を解く
sol = sp.solve([eq1, eq2, eq3], [a, b, c])

# 解を数値に変換
a_val, b_val, c_val = [float(sol[a]), float(sol[b]), float(sol[c])]

# グラフ描画関数
def plot_data(x, y, a, b, c):
    fig, ax = plt.subplots(dpi=100, figsize=(6, 6))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.scatter(x, y, marker='o', color='b')
    ax.plot(x, a * x**2 + b * x + c, ls='--', color='tab:red')
    ax.grid()
    plt.show()

# グラフを描画
plot_data(x_data, y_data, a_val, b_val, c_val)