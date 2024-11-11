import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    r1 = "weights_val/uav/angle"
    r2 = "weights_val/uav/bone"
    r3 = "weights_val/uav/joint"
    r4 = "weights_val/uav/motion"
    
    with open('../data/val_label.npy', 'rb') as f:
        label = np.load(f)

    with open(os.path.join(r1, 'best_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(r2, 'best_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())


    with open(os.path.join(r3, 'best_score.pkl'), 'rb') as r3:
        r3 = list(pickle.load(r3).items())

    with open(os.path.join(r4, 'best_score.pkl'), 'rb') as r4:
        r4 = list(pickle.load(r4).items())

    right_num = total_num = right_num_5 = 0
    best = 0.0

    total_num = 0
    right_num = 0

    alpha = [0.7,0.7,0.7,0.8]

    # 创建一个列表来存储融合结果
    fused_results = []

    for i in tqdm(range(len(label))):
        l = label[i]
        
        _, r11 = r1[i]
        _, r22 = r2[i]
        _, r33 = r3[i]
        _, r44 = r4[i]

        r = r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2] + r44 * alpha[3]

        # 将融合结果添加到列表中
        fused_results.append(r)

        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r_max = np.argmax(r)
        right_num += int(r_max == int(l))
        total_num += 1

    # 将融合结果列表转换为numpy数组
    fused_results_array = np.array(fused_results)

    # 保存融合结果为prey.npy文件
    np.save('pred.npy', fused_results_array)

    acc = right_num / total_num
    print(acc, alpha)
    if acc > best:
        best = acc
        best_alpha = alpha
    acc5 = right_num_5 / total_num

    print(best, best_alpha)
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
    print('Fusion results saved to prey.npy')
