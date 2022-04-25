import argparse
import moviepy
import numpy as np
import open3d as o3d
import os
from pathlib import Path
import shutil
from typing import List, Tuple

RANGE = np.array([[0, 70.4], [-40, 40]])
BLUE = [0, 0, 1]
RED = [1, 0, 0]

def clip_points(points: np.ndarray) -> np.ndarray:
    mask = (points[:, 0] >= RANGE[0, 0]) \
        & (points[:, 0] <= RANGE[0, 1]) \
        & (points[:, 1] >= RANGE[1, 0]) \
        & (points[:, 1] <= RANGE[1, 1])
    return points[mask]

def create_pcd(vec: o3d.utility.Vector3dVector) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud(vec)

def get_scenes(demo_root: Path, start: int, duration: int, validate: bool = True) -> List[int]:
    # val.txt contains all scenes for demo dataset
    scenes_file = demo_root / 'data' / 'lidar' / 'kitti_format' / 'ImageSets' / 'val.txt'
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

def get_geometries(demo_root: Path, scene: int) -> Tuple[
    o3d.utility.Vector3dVector, 
    o3d.utility.Vector3dVector
]:
    lidar_file = demo_root / 'data' / 'lidar' / 'kitti_format' / 'lidar' / f'{scene}.bin'
    lidar_points = clip_points(np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4))
    radar_file = demo_root / 'data' / 'radar' / 'kitti_format' / 'radar' / f'{scene}.bin'
    radar_points = clip_points(np.fromfile(radar_file, dtype=np.float32).reshape(-1, 6))
    return o3d.utility.Vector3dVector(lidar_points[:, :3]), \
        o3d.utility.Vector3dVector(radar_points[:, :3])

def produce_img(
    vis: o3d.visualization.Visualizer, 
    demo_root: Path, 
    scene: int, 
    lidar: o3d.geometry.PointCloud,
    radar: o3d.geometry.PointCloud
) -> None:
    tmp_lidar, tmp_radar = get_geometries(demo_root, scene)
    # lidar
    lidar.points = tmp_lidar
    lidar.paint_uniform_color(BLUE)
    vis.update_geometry(lidar)
    # radar
    radar.points = tmp_radar
    radar.paint_uniform_color(RED)
    vis.update_geometry(radar)
    # re-render
    vis.poll_events()
    vis.update_renderer()

def produce_imgs(demo_root: Path, section: List[int]) -> None:
    out_dir = demo_root / 'out'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920,height=1080)
    tmp_lidar, tmp_radar = get_geometries(demo_root, section[0])
    lidar_pcd = create_pcd(tmp_lidar)
    radar_pcd = create_pcd(tmp_radar)
    vis.add_geometry(lidar_pcd)
    vis.add_geometry(radar_pcd)
    vis.get_render_option().load_from_json(str(demo_root / 'render_options.json'))
    for scene in section:
        produce_img(vis, demo_root, scene, lidar_pcd, radar_pcd)

def make_demo(demo_root: Path, start: int, duration: int, fps: int) -> None:
    section = get_scenes(demo_root, start, duration)
    produce_imgs(demo_root, section)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('demo_root', help='path to root of directory containing point clouds, ground-truth and detections')
    parser.add_argument('start', help='timestamp of first scene')
    parser.add_argument('--duration', help='duration of movie')
    parser.add_argument('--fps', help='FPS of movie')
    args = parser.parse_args()
    args.duration   = 5 if args.duration is None else args.duration
    args.fps        = 10 if args.fps is None else args.fps
    make_demo(
        demo_root = Path(args.demo_root),
        start = int(args.start),
        duration = int(args.duration),
        fps = int(args.fps)
    )

if __name__ == '__main__':
    main()
