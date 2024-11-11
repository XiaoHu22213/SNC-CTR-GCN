#!/bin/bash

python main_gpu.py --weights work_dir/uav/joint_longtail/epoch_70_18270.pt --config config/uav/test/joint_longtail.yaml --phase test --work_dir weights/uav/joint_longtail

python main_gpu.py --weights work_dir/uav/bone_longtail/epoch_75_19575.pt --config config/uav/test/bone_longtail.yaml --phase test --work_dir weights/uav/bone_longtail

python main_gpu.py --weights work_dir/uav/joint_motion_SNC/epoch_70_18270.pt --config config/uav/test/joint_motion_SNC.yaml --phase test --work_dir weights/uav/joint_motion_SNC

