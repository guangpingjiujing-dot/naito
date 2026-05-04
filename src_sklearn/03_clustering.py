# ==============================================================
# 03_clustering.py  ―  クラスタリングの基本（K-Means）
# ==============================================================
# 【クラスタリングとは？】
#   正解ラベルなしで、似たデータをグループ（クラスタ）に分ける手法です。
#   「教師なし学習」と呼ばれ、データの傾向を発見するのに使います。
#   例：顧客を購買傾向でグループ分けするセグメンテーション
#
# 【K-Means とは？】
#   ① K 個のクラスタ中心をランダムに配置
#   ② 各データを最も近い中心に割り当て
#   ③ 各クラスタの平均を新しい中心にする
#   ④ ②③ を中心が動かなくなるまで繰り返す
# ==============================================================
# 実行方法: uv run 03_clustering.py
# ==============================================================

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams['font.family'] = 'sans-serif'

# ----------------------------------------------------------------
# 1. サンプルデータの生成
# ----------------------------------------------------------------
np.random.seed(42)

# make_blobs: クラスタ状の人工データを生成するユーティリティ
# n_samples    : 生成するデータ数
# centers      : クラスタの数（= 真のグループ数）
# cluster_std  : クラスタの広がり（大きいほど重なりやすい）
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

print("=== データの確認 ===")
print(f"サンプル数: {len(X)}")
print(f"特徴量の数: {X.shape[1]}  （2次元なので散布図で確認できる）")

# ----------------------------------------------------------------
# 2. 前処理：標準化
# ----------------------------------------------------------------
# 特徴量のスケール（単位）が異なる場合、距離計算が偏ってしまう
# StandardScaler で平均0・標準偏差1に揃える（= 標準化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------------
# 3. 最適なクラスタ数を探す（エルボー法）
# ----------------------------------------------------------------
# KMeans は「クラスタ数 K」を事前に指定する必要がある
# エルボー法: K を変えながら慣性（inertia）を計算し、
#             グラフが"折れ曲がる（肘のような）"点が最適 K の目安
inertias = []
k_range = range(1, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(k_range, inertias, marker="o", color="steelblue")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title("Elbow Method - Finding Optimal K")
plt.xticks(k_range)
plt.tight_layout()
plt.savefig("03_elbow.png", dpi=100)
plt.show()
print("エルボーグラフを 03_elbow.png に保存しました。")

# ----------------------------------------------------------------
# 4. K=4 でモデルを学習（エルボー法の結果に基づく）
# ----------------------------------------------------------------
best_k = 4
model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)   # 各サンプルのクラスタ番号を返す

# シルエットスコア: クラスタの"まとまり具合"を示す指標
# -1〜1 の範囲で、1 に近いほど質の高いクラスタリング
score = silhouette_score(X_scaled, labels)
print(f"\nシルエットスコア (K={best_k}): {score:.4f}  （1 に近いほど良い）")

# ----------------------------------------------------------------
# 5. クラスタリング結果の可視化
# ----------------------------------------------------------------
colors = ["steelblue", "tomato", "forestgreen", "gold"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左: 正解（真のグループ）
axes[0].set_title("True Groups (Ground Truth)")
for i in range(best_k):
    mask = y_true == i
    axes[0].scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.7, label=f"Group {i}")
axes[0].legend()

# 右: K-Means の予測結果
axes[1].set_title(f"K-Means Result (K={best_k})")
for i in range(best_k):
    mask = labels == i
    axes[1].scatter(
        X_scaled[mask, 0], X_scaled[mask, 1], c=colors[i], alpha=0.7, label=f"Cluster {i}"
    )
# クラスタ中心をプロット（大きい × マーク）
centers = model.cluster_centers_
axes[1].scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=200, zorder=5, label="Centroids")
axes[1].legend()

plt.tight_layout()
plt.savefig("03_clustering_result.png", dpi=100)
plt.show()
print("クラスタリング結果を 03_clustering_result.png に保存しました。")

# ----------------------------------------------------------------
# 6. 各クラスタのサンプル数を確認
# ----------------------------------------------------------------
print("\n=== 各クラスタのサンプル数 ===")
unique, counts = np.unique(labels, return_counts=True)
for cluster_id, count in zip(unique, counts):
    print(f"  Cluster {cluster_id}: {count} サンプル")
