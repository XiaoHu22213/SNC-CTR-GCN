import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
from itertools import product

def evaluate(alpha, r1, r2, r3, r4, r5, r6, label):
    right_num = total_num = right_num_5 = 0
    total_num = len(label)

    for i in range(total_num):
        l = label[i]

        _, r11 = r1[i]
        _, r22 = r2[i]
        _, r33 = r3[i]
        _, r44 = r4[i]
        _, r55 = r5[i]
        _, r66 = r6[i]

        r = r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2] + r44 * alpha[3] + r55 * alpha[4] + r66 * alpha[5]

        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r_max = np.argmax(r)
        right_num += int(r_max == int(l))

    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    return acc, acc5

if __name__ == "__main__":
    r1 = "weights_val/uav/joint"
    r2 = "weights_val/uav/joint_longtail"
    r3 = "weights_val/uav/bone_longtail"
    r4 = "weights_val/uav/joint_motion_SNC"
    r5 = "weights_val/uav/joint_SNC"
    r6 = "weights_val/uav/bone_SNC"

    print("Loading data...")
    with open('../data/val_label.npy', 'rb') as f:
        label = np.load(f)

    with open(os.path.join(r1, 'best_score.pkl'), 'rb') as f:
        r1 = list(pickle.load(f).items())
    with open(os.path.join(r2, 'best_score.pkl'), 'rb') as f:
        r2 = list(pickle.load(f).items())
    with open(os.path.join(r3, 'best_score.pkl'), 'rb') as f:
        r3 = list(pickle.load(f).items())
    with open(os.path.join(r4, 'best_score.pkl'), 'rb') as f:
        r4 = list(pickle.load(f).items())
    with open(os.path.join(r5, 'best_score.pkl'), 'rb') as f:
        r5 = list(pickle.load(f).items())
    with open(os.path.join(r6, 'best_score.pkl'), 'rb') as f:
        r6 = list(pickle.load(f).items())

    best_acc = 0.0
    best_acc5 = 0.0
    best_alpha = None

    num_iterations = 100000  # 设置迭代次数

    print("Starting random search...")
    for _ in tqdm(range(num_iterations), desc="Searching for best alpha"):
        # 随机生成 alpha 值
        alpha = np.round(np.random.uniform(0, 1, 6), 1)

        # 评估当前 alpha
        acc, acc5 = evaluate(alpha, r1, r2, r3, r4, r5, r6, label)

        # 更新最佳结果
        if acc > best_acc or (acc == best_acc and acc5 > best_acc5):
            best_acc = acc
            best_acc5 = acc5
            best_alpha = alpha

    print('Best alpha:', best_alpha)
    print('Best Top1 Acc: {:.4f}%'.format(best_acc * 100))
    print('Corresponding Top5 Acc: {:.4f}%'.format(best_acc5 * 100))

    print("Generating final predictions with best alpha...")
    # 使用最佳 alpha 生成最终预测结果
    fused_results = []
    for i in tqdm(range(len(label)), desc="Fusing results"):
        _, r11 = r1[i]
        _, r22 = r2[i]
        _, r33 = r3[i]
        _, r44 = r4[i]
        _, r55 = r5[i]
        _, r66 = r6[i]

        r = r11 * best_alpha[0] + r22 * best_alpha[1] + r33 * best_alpha[2] + r44 * best_alpha[3] + r55 * best_alpha[4] + r66 * best_alpha[5]
        fused_results.append(r)

    fused_results_array = np.array(fused_results)
    np.save('pred.npy', fused_results_array)
    print('Fusion results saved to pred.npy')
