import argparse
import time
import moviepy.video.io.ImageSequenceClip
import numpy as np
import open3d as o3d
import glob
import os
from pathlib import Path
import shutil
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R

RANGE = np.array([[0, 70.4], [-40, 40]])
BLACK = [0, 0, 0]
BLUE = [0, 0, 1]
RED = [1, 0, 0]
ORANGE = [1, 0.75, 0]
PINK = [0.9, 0.6, 0.6]

def clip_points(points: np.ndarray) -> np.ndarray:
    mask = (points[:, 0] >= RANGE[0, 0]) \
        & (points[:, 0] <= RANGE[0, 1]) \
        & (points[:, 1] >= RANGE[1, 0]) \
        & (points[:, 1] <= RANGE[1, 1])
    _points = points[mask]
    return np.column_stack((-_points[:, 1], _points[:, 0], _points[:, 2]))

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

def get_box(params: float, colour: List[int]) -> o3d.geometry.LineSet:
    def get_rotation(yaw):
        angle = np.array([0, 0, yaw])
        r = R.from_euler('XYZ', angle)
        return r.as_matrix()

    center = np.column_stack((params[0], params[1], params[2]))
    extent = params[3:6]  # h, w, l
    angle = params[6]
    # angle[0] = -angle[0]
    rot_m = get_rotation(angle)
    # rot_m = np.eye(3)
    bbox = o3d.geometry.OrientedBoundingBox(center.T, rot_m, extent.T)
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
    line_set.paint_uniform_color(colour)
    return line_set

def show_boxes(
    vis: o3d.visualization.Visualizer, 
    boxes: List[o3d.geometry.LineSet],
    new_boxes_path: Path,
    colour: List[int],
    remove_old: bool = True
) -> List[o3d.geometry.LineSet]:
    if remove_old:
        for box in boxes:
            vis.remove_geometry(box, reset_bounding_box=False)
    new_boxes = []
    boxes_params = np.fromfile(new_boxes_path, dtype=np.float32).reshape(-1, 7)
    for i in range(boxes_params.shape[0]):
        box = get_box(boxes_params[i], colour)
        new_boxes.append(box)
        vis.add_geometry(box, reset_bounding_box=(not remove_old))
    return new_boxes

def produce_img(
    vis: o3d.visualization.Visualizer, 
    demo_root: Path, 
    scene: int, 
    lidar_pcd: o3d.geometry.PointCloud,
    radar_pcd: o3d.geometry.PointCloud,
    gt_boxes: List[o3d.geometry.LineSet],
    lidar_boxes: List[o3d.geometry.LineSet],
    radar_boxes: List[o3d.geometry.LineSet]
) -> Tuple[List[o3d.geometry.LineSet], List[o3d.geometry.LineSet], List[o3d.geometry.LineSet]]:
    tmp_lidar, tmp_radar = get_geometries(demo_root, scene)
    # lidar
    lidar_pcd.points = tmp_lidar
    lidar_pcd.paint_uniform_color(BLUE)
    vis.update_geometry(lidar_pcd)
    # radar
    radar_pcd.points = tmp_radar
    radar_pcd.paint_uniform_color(RED)
    vis.update_geometry(radar_pcd)
    # bboxes
    gt_path = demo_root / 'data' / 'predictions' / 'lidar' / 'gt' / f'{scene}.txt'
    lidar_pred_path = demo_root / 'data' / 'predictions' / 'lidar' / 'pred' / f'{scene}.txt'
    radar_pred_path = demo_root / 'data' / 'predictions' / 'radar' / 'pred' / f'{scene}.txt'
    gt_boxes = show_boxes(vis, gt_boxes, gt_path, BLACK)
    lidar_boxes = show_boxes(vis, lidar_boxes, lidar_pred_path, ORANGE)
    radar_boxes = show_boxes(vis, radar_boxes, radar_pred_path, PINK)
    # re-render
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(demo_root / 'out' / f'{scene}.png'))
    return gt_boxes, lidar_boxes, radar_boxes

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
    boxes_dir = demo_root / 'data' / 'predictions'
    gt_boxes = show_boxes(vis, [], boxes_dir / 'lidar' / 'gt' / f'{section[0]}.txt', BLACK, remove_old=False)
    lidar_boxes = show_boxes(vis, [], boxes_dir / 'lidar' / 'pred' / f'{section[0]}.txt', ORANGE, remove_old=False)
    radar_boxes = show_boxes(vis, [], boxes_dir / 'radar' / 'pred' / f'{section[0]}.txt', PINK, remove_old=False)
    vis.add_geometry(lidar_pcd)
    vis.add_geometry(radar_pcd)
    vis.get_render_option().load_from_json(str(demo_root / 'render_options.json'))
    for scene in section:
        gt_boxes, lidar_boxes, radar_boxes = produce_img(
            vis, 
            demo_root, 
            scene, 
            lidar_pcd, 
            radar_pcd, 
            gt_boxes=gt_boxes, 
            lidar_boxes=lidar_boxes, 
            radar_boxes=radar_boxes)

def make_video(demo_root: Path, fps: int) -> None:
    images = sorted(glob.glob(str(demo_root / 'out' / '*.png')))
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=fps)
    clip.write_videofile(str(demo_root / 'out' / 'demo.mp4'))

def make_demo(demo_root: Path, starts: List[int], duration: int, fps: int) -> None:
    section = []
    for start in starts:
        section += get_scenes(demo_root, start, duration)
    produce_imgs(demo_root, section)
    make_video(demo_root, fps)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('demo_root', help='path to root of directory containing point clouds, ground-truth and detections')
    parser.add_argument('starts', help='timestamps of first scenes')
    parser.add_argument('--duration', help='duration of movie')
    parser.add_argument('--fps', help='FPS of movie')
    args = parser.parse_args()
    args.duration   = 5 if args.duration is None else args.duration
    args.fps        = 5 if args.fps is None else args.fps
    make_demo(
        demo_root = Path(args.demo_root),
        starts = [int(start) for start in args.starts.split(',')],
        duration = int(args.duration),
        fps = int(args.fps)
    )

if __name__ == '__main__':
    main()
