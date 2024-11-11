#!/bin/bash

python main_gpu.py --config config/uav/joint.yaml

python main_gpu.py --config config/uav/bone.yaml

python main_gpu.py --config config/uav/joint_motion.yaml

python main_gpu.py --config config/uav/bone_motion.yaml

python main_gpu.py --config config/uav/joint_SNC.yaml

python main_gpu.py --config config/uav/bone_SNC.yaml