#!/bin/bash

cd ~/mmdetection3d
source ~/miniconda3/etc/profile.d/conda.sh
conda activate open-mmlab
python tools/train.py configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py --work-dir .
