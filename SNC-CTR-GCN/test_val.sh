#!/bin/bash

python main_gpu.py --weights work_dir/uav/joint/epoch_70_18270.pt --config config/uav/joint.yaml --phase test --work_dir weights/uav/joint

python main_gpu.py --weights work_dir/uav/bone/epoch_75_19575.pt --config config/uav/bone.yaml --phase test --work_dir weights/uav/bone

python main_gpu.py --weights work_dir/uav/joint_motion/epoch_62_16182.pt --config config/uav/joint_motion.yaml --phase test --work_dir weights/uav/joint_motion

python main_gpu.py --weights work_dir/uav/bone_motion/epoch_73_19053.pt --config config/uav/bone_motion.yaml --phase test --work_dir weights/uav/bone_motion

python main_gpu.py --weights work_dir/uav/joint_SNC/epoch_78_20358.pt --config config/uav/joint_SNC.yaml --phase test --work_dir weights/uav/joint_SNC

python main_gpu.py --weights work_dir/uav/bone_SNC/epoch_64_16704.pt --config config/uav/bone_SNC.yaml --phase test --work_dir weights/uav/bone_SNC