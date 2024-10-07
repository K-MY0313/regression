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
xmin, xmax = 0, 50
ymin, ymax = 0, 2500
a, b = 40, 0


def init_graph(dpi=100, grid=True):
    fig, ax = plt.subplots(dpi=dpi, figsize=(6, 6))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if grid:
        ax.grid()
    return fig, ax


def plot_data(ax, D, offset_y=None, marker='o'):
    A = []
    for i, row in enumerate(D):
        A.append(ax.scatter(D[i, 0], D[i, 1], marker=marker))
        if offset_y is not None:
            A.append(ax.text(D[i, 0], D[i, 1] + offset_y[i], f'$(x_{i+1}, y_{i+1})$', ha='center'))
    return A


def draw_means(X, Y):
    xm = np.mean(X)
    ym = np.mean(Y)
    plt.vlines([xm], xmin, 1000, "black", linestyles='dashed', label=r"$\bar{x}$")
    plt.hlines([ym], ymin, ymax, "black", linestyles='dotted', label=r"$\bar{y}$")


def draw_regions(ax, X, Y):
    xm = np.mean(X)
    ym = np.mean(Y)
    ax.add_patch(
        matplotlib.patches.Rectangle((0, 0), xm, ym, facecolor='red', alpha=0.1, fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle((xm, ym), 50-xm, 3000-ym, facecolor='red', alpha=0.1, fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle((0, ym), xm, 3000-ym, facecolor='blue', alpha=0.1, fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle((xm, 0), 50-xm, ym, facecolor='blue', alpha=0.1, fill=True))


fig, ax = init_graph()
# a,b=38,0 の直線を追加
a_new = 39
b_new = -20
x_vals = np.array(ax.get_xlim())
y_vals = a_new * x_vals + b_new
ax.plot(x_vals, y_vals, '--', color='green', label=f'{a_new}*X+{b_new}')
plot_data(ax, D)
draw_means(X, Y)
draw_regions(ax, X, Y)
plt.legend(loc='upper left')

# Call plt.show() only once to display the plot
plt.show()