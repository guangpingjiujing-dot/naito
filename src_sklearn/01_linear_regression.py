# ==============================================================
# 01_linear_regression.py  ―  線形回帰の基本
# ==============================================================
# 【線形回帰とは？】
#   入力値（X）と出力値（y）の関係を直線（y = ax + b）で表し、
#   未知のデータに対して数値を予測する手法です。
#   例：「部屋の広さ」→「家賃」を予測する
# ==============================================================
# 実行方法: uv run 01_linear_regression.py
# ==============================================================

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 日本語フォントが無い環境でも文字化けしないように英語ラベルを使用
matplotlib.rcParams['font.family'] = 'sans-serif'

# ----------------------------------------------------------------
# 1. データの準備
# ----------------------------------------------------------------
# ランダムシードを固定することで、実行するたびに同じ結果が得られる
np.random.seed(42)

# 説明変数 X：部屋の広さ（㎡）を 50 件分ランダムに生成
X = np.random.rand(50, 1) * 50 + 10   # 10〜60㎡

# 目的変数 y：家賃（万円）= 0.8 × 広さ + 3 + ノイズ
y = 0.8 * X.flatten() + 3 + np.random.randn(50) * 2

print("=== データの確認 ===")
print(f"サンプル数: {len(X)}")
print(f"Xの範囲: {X.min():.1f} ～ {X.max():.1f} ㎡")
print(f"yの範囲: {y.min():.1f} ～ {y.max():.1f} 万円")

# ----------------------------------------------------------------
# 2. データの分割（学習用 / テスト用）
# ----------------------------------------------------------------
# モデルの性能を正しく評価するため、データを 80% 学習・20% テストに分ける
# test_size=0.2  → テストデータの割合
# random_state=42 → シードを固定して再現性を確保
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n学習データ数: {len(X_train)}, テストデータ数: {len(X_test)}")

# ----------------------------------------------------------------
# 3. モデルの学習（フィッティング）
# ----------------------------------------------------------------
# LinearRegression クラスのインスタンスを作成
model = LinearRegression()

# fit() でモデルに学習させる
# 内部で y = ax + b の a（傾き）と b（切片）を自動的に求める
model.fit(X_train, y_train)

print("\n=== 学習結果 ===")
print(f"傾き (a): {model.coef_[0]:.4f}")       # 広さが 1㎡ 増えると家賃がいくら上がるか
print(f"切片 (b): {model.intercept_:.4f}")      # X=0 のときの予測値（理論上の基準値）

# ----------------------------------------------------------------
# 4. 予測と評価
# ----------------------------------------------------------------
# テストデータに対して予測を行う
y_pred = model.predict(X_test)

# 評価指標を計算する
# MSE（平均二乗誤差）: 予測値と正解値の差を二乗して平均したもの（小さいほど良い）
mse = mean_squared_error(y_test, y_pred)

# R^2スコア: 1.0 が完璧、0.0 はただの平均値と同じ精度（大きいほど良い）
r2 = r2_score(y_test, y_pred)

print("\n=== モデルの評価 ===")
print(f"MSE  (平均二乗誤差): {mse:.4f}")
print(f"R^2スコア          : {r2:.4f}  （1.0 に近いほど精度が高い）")

# ----------------------------------------------------------------
# 5. 結果の可視化
# ----------------------------------------------------------------
plt.figure(figsize=(8, 5))

# 学習データを散布図で表示
plt.scatter(X_train, y_train, color="steelblue", alpha=0.6, label="Train data")

# テストデータを別の色で表示
plt.scatter(X_test, y_test, color="orange", alpha=0.8, label="Test data")

# 学習したモデルの回帰直線を描画
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color="red", linewidth=2, label="Regression line")

plt.xlabel("Room size (m²)")
plt.ylabel("Rent (10,000 JPY)")
plt.title("Linear Regression: Room Size vs Rent")
plt.legend()
plt.tight_layout()
plt.savefig("01_linear_regression.png", dpi=100)
plt.show()
print("\nグラフを 01_linear_regression.png に保存しました。")
