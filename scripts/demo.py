import argparse
import moviepy
import numpy as np
import open3d as o3d
import os
from pathlib import Path
import shutil
from typing import List


def get_scenes(demo_root: Path, start: int, duration: int, validate: bool = True) -> List[int]:
    # val.txt contains all scenes for demo dataset
    scenes_file = demo_root / 'data' / 'kitti_format' / 'ImageSets' / 'val.txt'
    with open(scenes_file, 'r') as fh:
        scenes = fh.readlines()
        scenes_num = [int(s.rstrip()) for s in scenes]
    end = start + duration * 1000
    section = [s for s in scenes_num if s >= start and s <= end]
    if validate:
        unexpected_jumps = [(0 if (b - a) <= 200 else 1) for (a, b) in zip(section[:-1], section[1:])]
        assert sum(unexpected_jumps) == 0, f'found {sum(unexpected_jumps)} unexpected jumps'
        assert len(section) == duration * 10 + 1, f'expected {duration * 10 + 1} scenes, found {len(section)}'
    return section

def produce_img(vis: o3d.visualization.Visualizer, lidar: Path, out: Path) -> None:
    lidar_pcd = np.fromfile(lidar, dtype=np.float32).reshape(-1, 4)
    print(1)
    pass

def produce_imgs(demo_root: Path, section: List[int]) -> None:
    out_dir = demo_root / 'out'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920,height=1080)
    for scene in section:
        lidar_file = demo_root / 'data' / 'kitti_format' / 'lidar' / f'{scene}.bin'
        out_file = out_dir / f'{scene}.png'
        produce_img(
            vis,
            lidar=lidar_file,
            out=out_file
        )

def make_demo(demo_root: Path, start: int, duration: int, fps: int, clip: bool) -> None:
    section = get_scenes(demo_root, start, duration)
    produce_imgs(demo_root, section)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('demo_root', help='path to root of directory containing point clouds, ground-truth and detections')
    parser.add_argument('start', help='timestamp of first scene')
    parser.add_argument('--duration', help='duration of movie')
    parser.add_argument('--fps', help='FPS of movie')
    parser.add_argument('--clip', action='store_true', help='clip point clouds and objects to detection range')
    args = parser.parse_args()
    args.duration   = 5 if args.duration is None else args.duration
    args.fps        = 10 if args.fps is None else args.fps
    make_demo(
        demo_root = Path(args.demo_root),
        start = int(args.start),
        duration = int(args.duration),
        fps = int(args.fps),
        clip = args.clip
    )

if __name__ == '__main__':
    main()
