import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colors
import matplotlib.animation
from IPython.display import HTML
import japanize_matplotlib

# JSONファイルからデータを読み込む
with open('ice.json', 'r') as f:
    data = json.load(f)

X = np.array(data['X'])
Y = np.array(data['Y'])
D = np.column_stack((X, Y))  # 2次元配列に変換
xmin, xmax = 0, 35
ymin, ymax = 0, 2000
a, b = 40, 0

def msr(a, b, X, Y):
    xm = np.mean(X)  # Xの平均を計算
    ym = np.mean(Y)  # Yの平均を計算
    x2m = np.mean(X ** 2)  # Xの二乗の平均を計算
    y2m = np.mean(Y ** 2)  # Yの二乗の平均を計算
    xym = np.mean(X * Y)  # XとYの積の平均を計算
    return x2m * (a ** 2) + b ** 2 + 2 * xm * a * b - 2 * xym * a - 2 * ym * b + y2m  # MSRを計算して返す

# メッシュグリッドを作成します。
N = 1000
A, B = np.meshgrid(np.linspace(-1000, 80, N), np.linspace(-100, 3000, N))

# MSRを計算します。
J = msr(A, B, X, Y)

# 図と軸を作成します。
fig, ax = plt.subplots(dpi=100)
ax.set_xlabel('$a$')  # x軸のラベルを設定
ax.set_ylabel('$b$')  # y軸のラベルを設定

# MSRのヒートマップを対数スケールで描画します。
mesh = ax.pcolormesh(A, B, J, norm=matplotlib.colors.LogNorm(vmin=J.min(), vmax=J.max()), shading='auto')
cbar = fig.colorbar(mesh)  # カラーバーを追加
cbar.set_label(r'$\hat{L}_{\mathcal{D}}(a,b)$')  # カラーバーのラベルを設定

# 等高線マップを描画します（小さい領域に線を描かないように損失値をクリップ）
ax.contour(
    A, B, np.clip(J, 1e-3, None), vmin=1, colors='tab:red', linewidths=0.5, linestyles='dashed',
    norm=matplotlib.colors.LogNorm(vmin=J.min(), vmax=J.max()))

# 図を表示します。
plt.show()
