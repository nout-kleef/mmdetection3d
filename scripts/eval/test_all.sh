#!/bin/bash
ARCHIVE="/mnt/12T/nout/archive"
function test() {
    set -x
    python tools/test.py "$ARCHIVE/$1/$2" "$ARCHIVE/$1/epoch_$3.pth" "--eval" "$4" 2>&1 | tee "$ARCHIVE/$1/test_result_frustum.txt"
    set +x
} 

# kitti
# test "dissvm/training_full" "hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py" "80" "kitti"
# lidar 3+0
# test "V2/lidar" "inhouse_lidar.py" "120" "inhouse"
# # lidar 3+1
# test "gpu1/intensity" "inhouse_lidar.py" "120" "inhouse"
# # radar 2+3
# test "V3/radar_unfiltered_bev" "inhouse_radar_bev.py" "120" "inhouse"
# radar 3+3
test "V3/radar_unfiltered" "conf.py" "120" "inhouse"
