# SNC-CTR-GCN

github地址：https://github.com/XiaoHu22213/SNC-CTR-GCN

数据集地址：通过网盘分享的文件：data
链接: https://pan.baidu.com/s/1eSOSQsEAlOWqXit-GJ2dWg?pwd=kep2 提取码: kep2

模型百度网盘地址： 通过网盘分享的文件：SNC-CTR-GCN
链接: https://pan.baidu.com/s/1G3fitjgluO-jyyqBaHRVnw?pwd=dbsp 提取码: dbsp

权重文件地址：通过网盘分享的文件：权重文件
链接: https://pan.baidu.com/s/1DRNX_yP6WXV3qBpQN_2iIQ?pwd=ph9s 提取码: ph9s

日志文件地址： 通过网盘分享的文件：日志
链接: https://pan.baidu.com/s/1-VKK3BrQjSVV7aZwoNVTHg?pwd=bxjk 提取码: bxjk


# 环境准备
项目基于SNC-CTR-GCN、SkateFormer、sttformer进行改进

python >=3.8
需要安装
pip install -e torchlight
pip install torchpack==0.0.3
pip install tensorboardX
pip install einops
pip install timm
pip install thop
# 数据集获取

国赛百度网盘：通过网盘分享的文件：11-12国赛内容
链接: https://pan.baidu.com/s/18NBRMwX03ijTCzFXx3h--w?pwd=e6ix 提取码: e6ix

1.百度网盘获取处理完后的数据集，数据集有两种，分别位于/data文件下的原数据集 以及 /data/angle文件下的角度数据集

2.data/angle文件下有get_angle.py文件，用于获取角度数据

# 模型训练

模型采用双模型融合，分别为SNC-CTR-GCN、SkateFormer、sttformer三个模型。

查看train.sh文件运行该文件训练模型：
    sh ./train.sh

# 测试新数据集

查看test_val.sh文件运行该文件获得评估得分:
    sh ./test_val.sh
查看test_test.sh文件运行该文件获得测试得分:
    sh ./test_test.sh

# 模型融合

eval集的融合：
python ensemble.py

test集的融合：
python ensemble_v2.py
