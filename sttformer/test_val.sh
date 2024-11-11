#!/bin/bash

python main_gpu.py --weights work_dir/uav/angle/angle.pt --config ./config/uav/angle.yaml --work_dir ./weights_val/uav/angle --run_mode test --save_score true

python main_gpu.py --weights work_dir/uav/bone/bone.pt --config ./config/uav/bone.yaml --work_dir ./weights_val/uav/bone --run_mode test --save_score true

python main_gpu.py --weights work_dir/uav/joint/joint.pt --config ./config/uav/joint.yaml --work_dir ./weights_val/uav/joint --run_mode test --save_score true

python main_gpu.py --weights work_dir/uav/motion/motion.pt --config ./config/uav/motion.yaml --work_dir ./weights_val/uav/motion --run_mode test --save_score true


