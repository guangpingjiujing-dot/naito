# ==============================================================
# 02_classification.py  ―  分類の基本（決定木）
# ==============================================================
# 【分類とは？】
#   入力データがどのグループ（クラス）に属するかを予測する手法です。
#   例：「花びらの長さ・幅」→「アヤメの種類（3種）」を判定する
#
# 【決定木（Decision Tree）とは？】
#   「花びら長さ > 2.5cm？」のような Yes/No の質問を繰り返して
#   クラスを絞り込む、直感的に理解しやすいアルゴリズムです。
# ==============================================================
# 実行方法: uv run 02_classification.py
# ==============================================================

import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

matplotlib.rcParams['font.family'] = 'sans-serif'

# ----------------------------------------------------------------
# 1. データの読み込み（アイリスデータセット）
# ----------------------------------------------------------------
# scikit-learn には練習用のデータセットが組み込まれている
# アイリス（アヤメ）データセット: 3 種類のアヤメを 150 件分記録したもの
iris = load_iris()

# iris.data   : 特徴量（花びら・がく片の長さ・幅、4列）
# iris.target : 正解ラベル（0=Setosa, 1=Versicolor, 2=Virginica）
X = iris.data    # shape: (150, 4)
y = iris.target  # shape: (150,)

print("=== アイリスデータセット ===")
print(f"サンプル数      : {X.shape[0]}")
print(f"特徴量の数      : {X.shape[1]}  {iris.feature_names}")
print(f"クラス（種類）  : {iris.target_names}")

# ----------------------------------------------------------------
# 2. 学習データとテストデータに分割
# ----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    # stratify=y を指定すると、各クラスの割合を保ったまま分割できる
)

print(f"\n学習データ数: {len(X_train)}, テストデータ数: {len(X_test)}")

# ----------------------------------------------------------------
# 3. モデルの学習
# ----------------------------------------------------------------
# max_depth: 木の深さの上限（大きすぎると過学習しやすくなる）
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# ----------------------------------------------------------------
# 4. 予測と評価
# ----------------------------------------------------------------
y_pred = model.predict(X_test)

# accuracy_score: 正解率（全サンプル中、正しく予測できた割合）
acc = accuracy_score(y_test, y_pred)
print(f"\n正解率（Accuracy）: {acc:.4f}  （1.0 が最高）")

# classification_report: クラスごとの詳細な評価指標
# precision: 「Aと予測した中で本当にAだった割合」
# recall   : 「本当にAのサンプルを正しくAと予測できた割合」
# f1-score : precision と recall の調和平均
print("\n=== 詳細レポート ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ----------------------------------------------------------------
# 5. 決定木の可視化
# ----------------------------------------------------------------
plt.figure(figsize=(14, 6))
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,        # クラスごとに色を付ける
    rounded=True,       # ノードを丸角にする
    fontsize=10,
)
plt.title("Decision Tree (max_depth=3) - Iris Dataset")
plt.tight_layout()
plt.savefig("02_classification_tree.png", dpi=100)
plt.show()
print("決定木を 02_classification_tree.png に保存しました。")

# ----------------------------------------------------------------
# 6. 特徴量の重要度を確認
# ----------------------------------------------------------------
print("\n=== 特徴量の重要度（高いほど分類に役立っている） ===")
importances = model.feature_importances_
for name, imp in zip(iris.feature_names, importances):
    bar = "█" * int(imp * 40)
    print(f"  {name:<30} {imp:.4f}  {bar}")
