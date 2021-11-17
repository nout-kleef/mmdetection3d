import subprocess

subprocess.call([
    "./tools/dist_train.sh",
    "configs/diss/diss-nus-3d.py",
    "1"                             # 1 GPU
])
