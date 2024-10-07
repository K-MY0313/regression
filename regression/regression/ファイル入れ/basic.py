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
a,b = 40, 0

def init_graph(dpi=100):
    fig, ax = plt.subplots(dpi=dpi)
    # データポイントのプロット
    ax.scatter(X, Y, marker='.')   
# トレンドラインのプロット
    # タイトルとラベルの設定
    ax.set_title('最高気温とアイスクリーム・シャーベットの支出額')
    ax.set_xlabel('最高気温の月平均（℃）')
    ax.set_ylabel('支出額（円）')
    plt.xticks(np.arange(0, 36, 5))  # x軸の目盛りを5℃刻みに
    plt.yticks(np.arange(0, 2250, 250))  # y軸の目盛りを250円刻みに
    # プロットの範囲設定
    ax.set_xlim(xmin,  xmax)
    ax.set_ylim(ymin, ymax)
    return fig, ax

def plot_data(ax, D, marker='.'):
    A = []
    for i, row in enumerate(D):
        A.append(ax.scatter(row[0], row[1], marker=marker))
    return A

def plot_line(ax, a, b, label=None, color='black', alpha=1.):
    if label is None:
        label = f'$y = {a}x + {b}$' if a != 1 else f'$y = x + {b}$'
    x = np.array([xmin, xmax])
    return ax.plot(x, a * x + b, color, ls='-', label=label, alpha=alpha)

def plot_hat_y(ax, D, a, b, marker='*', show_text=True):
    A = []
    for i, row in enumerate(D):
        x = D[i,0]
        hat_y = a * x + b
        A.append(ax.scatter(x, hat_y, marker=marker))
       
    return A

def plot_error(plt, ax, D, a, b,):
    A = []
    for i, row in enumerate(D):
        x, y = row
        y_hat = a * x + b
        ymin = min(y, y_hat)
        ymax = max(y, y_hat)
        A.append(plt.vlines([x], ymin, ymax, linestyles='dashed', alpha=0.5))
        
    return A
# グリッドの表示
fig, ax = init_graph()
plot_data(ax, D)
plot_line(ax, a, b)
plot_hat_y(ax, D, a, b)
plot_error(plt, ax, D, a, b,)
# プロットの表示
plt.show()