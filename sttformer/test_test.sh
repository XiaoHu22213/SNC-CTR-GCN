#!/bin/bash

python main_gpu.py --weights work_dir/uav/angle/angle.pt --config ./config/uav/test/angle.yaml --work_dir ./weights_test/uav/angle --run_mode test --save_score true

python main_gpu.py --weights work_dir/uav/bone/bone.pt --config ./config/uav/test/bone.yaml --work_dir ./weights_test/uav/bone --run_mode test --save_score true

python main_gpu.py --weights work_dir/uav/joint/joint.pt --config ./config/uav/test/joint.yaml --work_dir ./weights_test/uav/joint --run_mode test --save_score true

python main_gpu.py --weights work_dir/uav/motion/motion.pt --config ./config/uav/test/motion.yaml --work_dir ./weights_test/uav/motion --run_mode test --save_score true


