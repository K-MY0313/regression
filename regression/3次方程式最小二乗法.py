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

X_matrix = np.vstack([np.ones(len(X)), X, X**2, X**3]).T#3次方程式最小二乗法
Y_vector = Y.reshape(-1, 1)
beta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ Y_vector
a, b, c, d = beta.flatten()

def init_graph(dpi=100, grid=True):
    fig, ax = plt.subplots(dpi=dpi, figsize=(6, 6))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.plot(X, d*X**3 + c*X**2 + b*X + a, ls='--', color='tab:red')
    if grid:
        ax.grid()
    return fig, ax

fig, ax = init_graph()
ax.scatter(X, Y, marker='o', color='b')

# Call plt.show() only once to display the plot
plt.show()