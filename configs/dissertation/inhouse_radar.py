_base_ = 'pointpillars_inhouse-3class_radar.py'

# Model setup
data_root = 'data/inhouse_filtered_radar/kitti_format/'

# Experiment settings
runner = dict(max_epochs=160)
evaluation = dict(interval=1)

data = dict(samples_per_gpu=11, workers_per_gpu=4)