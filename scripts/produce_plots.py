from dataclasses import dataclass
import subprocess
import os
from typing import List

@dataclass
class Experiment:
    name: str
    path: str
    files: List[str]

    @property
    def paths(self):
        return [os.path.join(self.path, f) for f in self.files]

def val_plots(exp1: Experiment, exp2: Experiment, metric: str, title: str, out_file: str):
    subprocess.check_output([
        'python',
        'tools/analysis_tools/analyze_logs.py',
        'plot_curve',
        *exp1.paths,
        *exp2.paths,
        '--title', title,
        '--keys', metric,
        '--legend', exp1.name, exp2.name,
        '--mode', 'eval',
        '--interval', '4',
        '--out', out_file,
        '--style', 'whitegrid'
    ])

def train_loss(exp1: Experiment, exp2: Experiment, multiclass: bool, title: str, out_file: str):
    metrics = [' loss_bbox', ' loss_cls'] if multiclass else ['']
    legend = [f'{e.name}{m}' for m in metrics for e in [exp1, exp2]]
    metrics = ['loss_bbox'] if not multiclass else ['loss_bbox', 'loss_cls']
    subprocess.check_output([
        'python',
        'tools/analysis_tools/analyze_logs.py',
        'plot_curve',
        *exp1.paths,
        *exp2.paths,
        '--title', title,
        '--keys', *metrics,
        '--legend', *legend,
        '--out', out_file,
        '--style', 'whitegrid'
    ])

def get_title(experiment_name: str, metric: str, diff_dist: bool) -> str:
    parts = metric.split('_')
    assert parts[1] == '3D'
    
    _class = parts[0].lower()
    if _class == 'overall':
        _class = 'mAP'
    else:
        _class = '$\\mathregular{AP_{' + _class + '}}$'
    _dist = parts[2]
    _iou = ' '.join(parts[3:])

    if diff_dist:
        return f'{experiment_name} - {_class}, {_dist} ({_iou})'
    else:
        assert _dist == 'far'
        return f'{experiment_name} - {_class} ({_iou})'


def experiment(experiment_name: str, vary: str, e1: Experiment, e2: Experiment):
    print(f'experiment: {experiment_name}')
    out_dir = f'/Users/nout/Documents/university/Dissertation/report/ug/images/plots/{experiment_name}'
    assert vary in {'dist', 'class'}
    DISTS = ['close', 'medium', 'far']
    CLSSS = ['overall', 'Car', 'Cyclist', 'Pedestrian']
    IOUS = ['strict', 'loose', 'very_loose']
    if vary == 'dist':
        metrics = [f'KITTI/Car_3D_{d}_{iou}' for d in DISTS for iou in IOUS]
    else:
        metrics = [f'KITTI/{c}_3D_far_{iou}' for c in CLSSS for iou in IOUS]
    try:
        out_file = os.path.join(out_dir, 'train_loss.pdf')
        title = f'{experiment_name} - train loss'
        train_loss(e1, e2, False, title, out_file)
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise
    for metric in metrics:
        assert metric.startswith('KITTI/')
        _metric = metric[6:]
        title = get_title(experiment_name, _metric, diff_dist=True)
        out_file = os.path.join(out_dir, f'{_metric}.pdf')
        try:
            val_plots(e1, e2, metric, title, out_file)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise

def main():
    experiment(
        experiment_name='LiDAR and radar',
        vary='dist',
        e1=Experiment('LiDAR', 'experiments/gpu1/intensity', ['20220323_140835.log.json','20220325_121729.log.json']),
        e2=Experiment('radar', 'experiments/V3/radar_unfiltered', ['20220322_082149.log.json']),
    )
    experiment(
        experiment_name='radar',
        vary='dist',
        e1=Experiment('w/ vertical dimension', 'experiments/V3/radar_unfiltered', ['20220322_082149.log.json']),
        e2=Experiment('w/o vertical dimension', 'experiments/V3/radar_unfiltered_bev', ['20220324_215623.log.json']),
    )
    experiment(
        experiment_name='LiDAR',
        vary='class',
        e1=Experiment('w/ intensity', 'experiments/gpu1/intensity', ['20220323_140835.log.json','20220325_121729.log.json']),
        e2=Experiment('w/o intensity', 'experiments/V2/lidar', ['20220316_162405.log.json']),
    )

if __name__ == '__main__':
    main()
