_base_ = 'pointpillars_inhouse-3class.py'

runner = dict(max_epochs=8)
evaluation = dict(interval=1)

data = dict(
        samples_per_gpu=8,
        workers_per_gpu=4)
