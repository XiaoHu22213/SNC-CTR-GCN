#!/bin/bash

python main_gpu.py --config ./config/uav/angle.yaml --work_dir ./work_dir/uav/angle

python main_gpu.py --config ./config/uav/bone.yaml --work_dir ./work_dir/uav/bone

python main_gpu.py --config ./config/uav/joint.yaml --work_dir ./work_dir/uav/joint

python main_gpu.py --config ./config/uav/motion.yaml --work_dir ./work_dir/uav/motion


