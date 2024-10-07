import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# JSONファイルを読み込む
with open('california_housing.json', 'r') as file:
    data = json.load(file)

# データフレームに変換
df = pd.DataFrame(data)

# 'Price' をターゲットに設定し、それ以外を特徴量とする
X = df.drop('Price', axis=1)
y = df['Price']

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train に定数項（1の列）を追加し、行列の形状を調整
X_b_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]

# y_train を列ベクトルに変換
y_train = y_train.values.reshape(-1, 1)

# 最小二乗法による回帰係数の計算
theta = np.linalg.inv(X_b_train.T.dot(X_b_train)).dot(X_b_train.T).dot(y_train)

# 特徴量ごとの係数を出力
print("回帰係数:\n", theta)

# テストデータにも定数項（1の列）を追加
X_b_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# 予測値を計算
predicted_prices = X_b_test.dot(theta)

# 結果を表示
predicted_prices = predicted_prices.flatten()  # 1次元に変換
for i in range(len(predicted_prices)):
    print(f"Predicted Price: {predicted_prices[i]}, Actual Price: {y_test.iloc[i]}, Error: {abs(y_test.iloc[i] - predicted_prices[i])}")
# 残差を計算
residuals = y_test.values - predicted_prices
average_error = np.mean(np.abs(predicted_prices - y_test.values))
print(f"平均残差: {average_error}")
# 残差プロット
# 残差プロット
plt.scatter(predicted_prices, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")

# 平均残差のテキストを追加
plt.text(0.5, 0.9, f"Average Residual: {average_error:.2f}", transform=plt.gca().transAxes)  # グラフの座標系で指定

plt.show()
# 平均残差を計算して表示

# 予測値 vs. 実測値プロット
plt.scatter(predicted_prices, y_test.values)
plt.plot([min(predicted_prices), max(predicted_prices)], [min(predicted_prices), max(predicted_prices)], color='r')
plt.xlabel("Predicted Price")
plt.ylabel("Actual Price")
plt.title("Predicted vs. Actual Prices")
plt.show()