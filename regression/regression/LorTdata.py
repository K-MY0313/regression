import json
import random

# JSONファイルからデータを読み込む
with open('ice.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

X = data['X']
Y = data['Y']

# データをシャッフルするためのインデックスリストを作成
indices = list(range(len(X)))
random.shuffle(indices)

# 80%のデータを訓練用、20%をテスト用に分割
split_point = int(len(X) * 0.8)

# データを分割
X_train = [X[i] for i in indices[:split_point]]
Y_train = [Y[i] for i in indices[:split_point]]
X_test = [X[i] for i in indices[split_point:]]
Y_test = [Y[i] for i in indices[split_point:]]

# 分割したデータを新しいJSONファイルに保存
train_data = {"X": X_train, "Y": Y_train}
test_data = {"X": X_test, "Y": Y_test}

with open('train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open('test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

# 分割されたデータの表示
print("訓練用データ:")
print(f"X_train (最初の5個): {X_train[:5]}")
print(f"Y_train (最初の5個): {Y_train[:5]}")
print(f"訓練用データの長さ: {len(X_train)}")

print("\nテスト用データ:")
print(f"X_test (最初の5個): {X_test[:5]}")
print(f"Y_test (最初の5個): {Y_test[:5]}")
print(f"テスト用データの長さ: {len(X_test)}")