import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class ModalityConfig:
    """模态配置类"""
    name: str
    path: str
    data: List = None  # 用于存储加载的数据

class EnsembleEvaluator:
    def __init__(self, modality_configs: List[ModalityConfig], label_path: str):
        self.modality_configs = modality_configs
        self.num_modalities = len(modality_configs)
        self.label = self._load_label(label_path)
        self._load_all_modalities()

    def _load_label(self, label_path: str) -> np.ndarray:
        """加载标签数据"""
        with open(label_path, 'rb') as f:
            return np.load(f)

    def _load_all_modalities(self):
        """加载所有模态的数据"""
        for config in self.modality_configs:
            with open(os.path.join(config.path, 'best_score.pkl'), 'rb') as f:
                config.data = list(pickle.load(f).items())

    def evaluate(self, alpha: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """评估函数"""
        right_num = right_num_5 = total_num = 0
        fused_results = []

        for i in range(len(self.label)):
            # 获取所有模态的预测结果
            predictions = [config.data[i][1] for config in self.modality_configs]

            # 计算加权融合结果
            result = sum(pred * w for pred, w in zip(predictions, alpha))
            fused_results.append(result)

            # 计算准确率
            l = self.label[i]
            rank_5 = result.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r_max = np.argmax(result)
            right_num += int(r_max == int(l))
            total_num += 1

        acc = right_num / total_num
        acc5 = right_num_5 / total_num
        return acc, acc5, np.array(fused_results)

def process_batch(args: Tuple[np.ndarray, EnsembleEvaluator]) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """处理单个批次的评估"""
    alpha, evaluator = args
    acc, acc5, fused_results = evaluator.evaluate(alpha)
    return acc, acc5, fused_results, alpha

def main():
    # 定义模态配置
    modality_configs = [
        ModalityConfig("joint", "SNC-CTR-GCN/weights_test/uav/joint"),                   #1.单48.45 67.60
        ModalityConfig("bone", "SNC-CTR-GCN/weights_test/uav/bone"),                     #1+2= 49.75      68.15
        ModalityConfig("joint_SNC", "SNC-CTR-GCN/weights_test/uav/joint_SNC"),           #1+4= 50.20 68.30
        ModalityConfig("bone_SNC", "SNC-CTR-GCN/weights_test/uav/bone_SNC"),             #1+4+5 = 50.60 68.85
        ModalityConfig("joint_motion", "SNC-CTR-GCN/weights_test/uav/joint_motion"),     #1+4+5+6 = 50.75  68.30
        ModalityConfig("bone_motion", "SNC-CTR-GCN/weights_test/uav/bone_motion"),
        ModalityConfig("angle", "SNC-CTR-GCN/weights_test/uav/angle"),
        ModalityConfig("SkateFormer", "SkateFormer/weights_test/uav/joint"),
        ModalityConfig("SkateFormer", "SkateFormer/weights_test/uav/bone"),
        # ModalityConfig("SkateFormer", "SkateFormer/weights_test/uav/joint_motion"),
        ModalityConfig("sttformer", "sttformer/weights_test/uav/angle"),
        ModalityConfig("sttformer", "sttformer/weights_test/uav/bone"),
        ModalityConfig("sttformer", "sttformer/weights_test/uav/joint"),
        ModalityConfig("sttformer", "sttformer/weights_test/uav/motion"),
    ]

    # 创建评估器
    evaluator = EnsembleEvaluator(modality_configs, 'data/test_label.npy')

    # 并行处理参数
    num_iterations = 1
    num_processes = 1

    # 准备参数
    all_params = []
    for _ in range(num_iterations):

        alpha = [1.0 ,0.0 ,1.0    ,0.0    , 0.137   ,0.5937   , 0.5047   ,1.0      , 0.9374   , 0.4302 , 0.0      , 0.8823   ,0.2424    ]
        all_params.append((alpha, evaluator))

    # 并行处理
    best_acc = 0.0
    best_acc5 = 0.0
    best_alpha = None
    best_fused_results = None

    print(f"Starting parallel processing with {num_processes} processes...")
    with mp.Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap(process_batch, all_params),
                           total=num_iterations,
                           desc="Processing batches"):
            acc, acc5, fused_results, alpha = result

            if best_acc > 0.515 or best_acc5 > 0.69:
                print('Best alpha:', best_alpha)
                print('Best Top1 Acc: {:.4f}%'.format(best_acc * 100))
                print('Corresponding Top5 Acc: {:.4f}%'.format(best_acc5 * 100))

            if acc > best_acc:
                best_acc = acc
                best_acc5 = acc5
                best_alpha = alpha
                best_fused_results = fused_results

    # 打印最终结果
    print('Best alpha:', best_alpha)
    print('Best Top1 Acc: {:.4f}%'.format(best_acc * 100))
    print('Corresponding Top5 Acc: {:.4f}%'.format(best_acc5 * 100))

    # 保存结果
    np.save('pred.npy', best_fused_results)
    print('Best fusion results saved to pred.npy')

if __name__ == "__main__":
    main()
