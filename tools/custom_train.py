import subprocess

subprocess.call([
    "./tools/dist_train.sh",
    "configs/diss/diss-nu-3d.py"
    "1"                             # 1 GPU
])
