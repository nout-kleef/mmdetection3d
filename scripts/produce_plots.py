from dataclasses import dataclass
import subprocess
import os

@dataclass
class Experiment:
    name: str
    path: str
    file: str

    @property
    def full_path(self):
        return os.path.join(self.path, self.file)

def val_plots(exp1: Experiment, exp2: Experiment, metric: str, title: str, out_file: str):
    subprocess.check_output([
        'python',
        'tools/analysis_tools/analyze_logs.py',
        'plot_curve',
        exp1.full_path,
        exp2.full_path,
        '--title', title,
        '--keys', metric,
        '--legend', exp1.name, exp2.name,
        '--mode', 'eval',
        '--interval', '4',
        '--out', out_file
    ])

def get_title(experiment_name: str, metric: str, diff_dist: bool) -> str:
    parts = metric.split('_')
    assert parts[1] == '3D'
    
    _class = parts[0].lower()
    _dist = parts[2]
    _iou = '_'.join(parts[3:])

    if diff_dist:
        return f'{experiment_name} - {_class}, {_dist} ({_iou})'
    else:
        assert _dist == 'far'
        return f'{experiment_name} - {_class} ({_iou})'


def main():
    experiment_name = 'radar'
    out_dir = f'/Users/nout/Documents/university/Dissertation/report/ug/images/plots/{experiment_name}'
    metrics = [
        'KITTI/Car_3D_close_loose', 
        'KITTI/Car_3D_medium_loose',
        'KITTI/Car_3D_far_loose',
    ]
    exp1 = Experiment(
        name='3D', 
        path='experiments/V3/radar_unfiltered', 
        file='20220322_082149.log.json'
    )
    exp2 = Experiment(
        name='2D', 
        path='experiments/V3/radar_unfiltered_bev', 
        file='20220324_215623.log.json'
    )
    for metric in metrics:
        assert metric.startswith('KITTI/')
        _metric = metric[6:]
        title = get_title(experiment_name, _metric, diff_dist=True)
        out_file = os.path.join(out_dir, f'{_metric}.pdf')
        try:
            val_plots(exp1, exp2, metric, title, out_file)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise

if __name__ == '__main__':
    main()
