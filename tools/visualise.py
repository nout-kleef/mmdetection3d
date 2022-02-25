import argparse
import os
import numpy as np
import open3d
from scipy.spatial.transform import Rotation as R


LIDAR_DTYPE = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')]
RADAR_DTYPE = [
    ('id', 'u4'),
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('vx', 'f4'), ('vy', 'f4'),
    ('fPower', 'f4'), ('fRCS', 'f4'), ('fSpeed', 'f4')
]
GT_DTYPE = dtype=[
    ('class', 'U8'),
    ('truncated', 'i1'), ('occluded', 'i1'), ('alpha', 'i1'),
    ('bbox0', 'i1'), ('bbox1', 'i1'), ('bbox2', 'i1'), ('bbox3', 'i1'),
    ('h', 'f4'), ('w', 'f4'), ('l', 'f4'),
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('yaw', 'f4')
]
LIDAR_EXT = [0, 0, -0.3, -2.5, 0, 0]
RADAR_EXT = [0.06, -0.2, 0.7, -3.5, 2, 180]

def load_pointcloud(file, dtype, transform, color):
    points_structured = np.fromfile(file, dtype=dtype)
    points = np.column_stack((
        points_structured['x'],
        points_structured['y'],
        points_structured['z'],
    ))
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    pcd.transform(transform)
    return pcd

def load_gt(file):
    boxes = []
    gt_data = np.loadtxt(file, dtype=GT_DTYPE, ndmin=1)
    for gt_obj in gt_data:
        center = np.column_stack((gt_obj['x'], gt_obj['y'], gt_obj['z']))
        extent = np.column_stack((gt_obj['l'], gt_obj['w'], gt_obj['h']))
        rot = get_rotation(gt_obj['yaw'])
        box = open3d.geometry.OrientedBoundingBox(center.T, rot, extent.T)
        boxes.append(box)
    return boxes

def visualise(args, ts, lidar_transform, radar_transform):
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=1920,height=1080)
    # lidar
    lidar_file = os.path.join(args.load_dir, 'lidar', f'{ts}.bin')
    lidar_pc = load_pointcloud(lidar_file, LIDAR_DTYPE, lidar_transform, [0, 0, 1])
    # radar
    radar_file = os.path.join(args.load_dir, 'radar', f'{ts}.bin')
    radar_pc = load_pointcloud(radar_file, RADAR_DTYPE, radar_transform, [1, 0, 0])
    # gt
    gt_file = os.path.join(args.load_dir, 'label', f'{ts}.txt')
    boxes = load_gt(gt_file)
    # display
    vis.add_geometry(lidar_pc)
    vis.add_geometry(radar_pc)
    for box in boxes:
        vis.add_geometry(box)
    vis.run()

def main():
    parser = argparse.ArgumentParser(description='Visualise lidar + radar pointclouds')
    parser.add_argument('--load_dir', help='Directory containing data')
    parser.add_argument('--save_dir', help='Directory used to save images (optional)')
    parser.add_argument('timestamps', nargs='*')
    args = parser.parse_args()

    save_img = args.save_dir is not None
    if save_img:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        else:
            os.system("rm -r " + args.save_dir)
            os.mkdir(args.save_dir)

    lidar_tr = get_matrix_from_ext(LIDAR_EXT)
    radar_tr = get_matrix_from_ext(RADAR_EXT)
    
    for ts in args.timestamps:
        visualise(args, ts, lidar_tr, radar_tr)

def get_matrix_from_ext(ext):
    rot = R.from_euler('ZYX', ext[3:], degrees=True)
    rot_m = rot.as_matrix()
    x, y, z = ext[:3]
    tr = np.eye(4)
    tr[:3,:3] = rot_m
    tr[:3, 3] = np.array([x, y, z]).T
    return tr

def get_rotation(yaw):
    angle = np.array([0, 0, yaw])
    r = R.from_euler('XYZ', angle)
    return r.as_matrix()

if __name__ == '__main__':
    main()
