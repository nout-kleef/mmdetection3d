_base_ = 'pointpillars_inhouse-3class.py'

runner = dict(max_epochs=180)
evaluation = dict(interval=4)

data = dict(
        samples_per_gpu=11,
        workers_per_gpu=4)
