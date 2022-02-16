_base_ = '../pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'

runner = dict(max_epochs=2)
evaluation = dict(interval=1)

data = dict(
        samples_per_gpu=12,
        workers_per_gpu=4)
