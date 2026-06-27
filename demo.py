import sys, os
sys.path.append(os.pardir)
import matplotlib.pyplot as plt

import numpy as np
from mnist import load_mnist
from two_layer_net2 import TwoLayerNet
from common.functions import softmax

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

# これはClaudeに生成してもらった
def show_batch(network, x, t, start, n=10):
    """ start 番目から n 枚を予測し、y を出力してから画像を並べて表示 """
    x_batch = x[start:start + n]
    t_batch = t[start:start + n]

    y = network.predict(x_batch)           # 生スコア (n, 10)
    prob = softmax(y)                       # 確率に変換（見やすさのため）
    pred = np.argmax(y, axis=1)            # 予測ラベル
    true = np.argmax(t_batch, axis=1) if t_batch.ndim != 1 else t_batch

    # --- y（数値）を先に出力 ---
    print(f"\n===== サンプル {start} 〜 {start + n - 1} =====")
    for i in range(len(x_batch)):
        mark = "○" if pred[i] == true[i] else "×"
        print(f"[{start + i:>5}] pred={pred[i]} true={true[i]} "
              f"conf={prob[i][pred[i]]:.2f} {mark}")

    # --- 画像をまとめて表示 ---
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.4))
    axes = np.atleast_1d(axes).flatten()
    for i in range(len(x_batch)):
        axes[i].imshow(x_batch[i].reshape(28, 28), cmap='gray')
        ok = pred[i] == true[i]
        axes[i].set_title(f"pred:{pred[i]} / true:{true[i]}",
                          color=('green' if ok else 'red'), fontsize=10)
        axes[i].axis('off')
    for j in range(len(x_batch), len(axes)):  # 余ったマスを消す
        axes[j].axis('off')
    fig.tight_layout()
    plt.show()  # ウィンドウを閉じるまでここで待機

    return pred, true


iters_num = 2000
batch_size = 100
lr = 0.1
train_size = x_train.shape[0]

for i in range(iters_num):
    idx = np.random.choice(train_size, batch_size)
    x_b, t_b = x_train[idx], t_train[idx]
    grad = network.gradient(x_b, t_b)   # ← 先に gradient() の grads={} 抜けを修正しておくこと
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= lr * grad[key]
    print('accuracy:', network.accuracy(x_test, t_test))

# ===== メインループ：10枚ずつ確認 =====
total = 50  # 確認したい枚数（適宜変更）
for start in range(0, total, 10):
    show_batch(network, x_test, t_test, start, n=10)
    cmd = input("Enter で次の10枚 / q で終了 > ").strip().lower()
    if cmd == 'q':
        break
    else:
        continue