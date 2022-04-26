#!/bin/bash

python tools/test.py tmp/radar.py /mnt/12T/nout/archive/V3/radar_unfiltered/epoch_120.pth \
--eval inhouse \
--eval-options show=True out_dir=/mnt/12T/nout/demo/data/predictions/radar
echo "FLAG"

python tools/test.py tmp/lidar.py \
/mnt/12T/nout/archive/gpu1/intensity/epoch_120.pth \
--eval inhouse \
--eval-options show=True out_dir=/mnt/12T/nout/demo/data/predictions/lidar
echo "FLAG"
