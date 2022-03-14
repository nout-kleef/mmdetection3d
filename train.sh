#!/bin/bash

sleep 3
echo "launching."

CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/dissertation/debug/nC_nT_nP.py --work-dir /mnt/12T/nout/archive/debug/nC_nT_nP
