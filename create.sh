#!/bin/bash

git checkout nC_T_P
python tools/create_data.py inhouse --root-path /mnt/12T/nout/inhouse_filtered --out-dir /mnt/12T/nout/debug/nC_T_P --workers 64 --extra-tag inhouse

git checkout C_nT_P
python tools/create_data.py inhouse --root-path /mnt/12T/nout/inhouse_filtered --out-dir /mnt/12T/nout/debug/C_nT_P --workers 64 --extra-tag inhouse

git checkout C_T_nP
python tools/create_data.py inhouse --root-path /mnt/12T/nout/inhouse_filtered --out-dir /mnt/12T/nout/debug/C_T_nP --workers 64 --extra-tag inhouse
