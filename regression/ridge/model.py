import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 学習データ
X = np.array([0.0, 0.16, 0.22, 0.34, 0.44, 0.5, 0.67, 0.73, 0.9, 1.0])
Y = np.array([-0.06, 0.94, 0.97, 0.85, 0.25, 0.09, -0.9, -0.93, -0.53, 0.08])

# 9次関数の特徴量を作成するためのオブジェクト
poly = PolynomialFeatures(degree=9)

# 正則化パラメータのリスト
alphas = [1e-9, 1e-6, 1e-3, 1]

# プロットの準備
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='black', label='データポイント')

# 各正則化パラメータでリッジ回帰を実行
for alpha in alphas:
    # 特徴量変換とリッジ回帰をパイプラインで繋げる
    model = make_pipeline(poly, Ridge(alpha=alpha))
    model.fit(X[:, np.newaxis], Y)
    X_plot = np.linspace(0, 1, 100)
    Y_plot = model.predict(X_plot[:, np.newaxis])
    plt.plot(X_plot, Y_plot, label=f'alpha={alpha}')
for alpha in alphas:
    model = make_pipeline(poly, Ridge(alpha=alpha))
    model.fit(X[:, np.newaxis], Y)

    # リッジ回帰モデルのパラメータを取得
    coef = model.steps[1][1].coef_

    # L2ノルムを計算
    l2_norm = np.linalg.norm(coef)

    print(f"alpha={alpha}, L2ノルム={l2_norm}")
# グラフの設定
plt.xlabel('X')
plt.ylabel('Y')
plt.title('9次関数によるリッジ回帰')
plt.legend()
plt.show()