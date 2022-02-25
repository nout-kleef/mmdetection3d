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

def visualise(args, ts, lidar_transform, radar_transform):
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=1920,height=1080)
    # lidar
    lidar_file = os.path.join(args.load_dir, 'lidar', f'{ts}.bin')
    lidar_pc = load_pointcloud(lidar_file, LIDAR_DTYPE, lidar_transform, [0, 0, 1])
    # radar
    radar_file = os.path.join(args.load_dir, 'radar', f'{ts}.bin')
    radar_pc = load_pointcloud(radar_file, RADAR_DTYPE, radar_transform, [1, 0, 0])
    # display
    vis.add_geometry(lidar_pc)
    vis.add_geometry(radar_pc)
    vis.run()
    # save_view_point(vis, os.path.join(args.save_dir, 'viewpoints', f'vp_{ts}.json'))

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
        if save_img:
            fname = os.path.join(args.save_dir, f'{ts}.png')
            # vis.capture_screen_image(fname)

def get_matrix_from_ext(ext):
    rot = R.from_euler('ZYX', ext[3:], degrees=True)
    rot_m = rot.as_matrix()
    x, y, z = ext[:3]
    tr = np.eye(4)
    tr[:3,:3] = rot_m
    tr[:3, 3] = np.array([x, y, z]).T
    return tr

def save_view_point(vis, filename):
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # open3d.io.write_pinhole_camera_parameters(filename, param)

def load_view_point(vis, filename):
    param = open3d.io.read_pinhole_camera_parameters(filename)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param)

if __name__ == '__main__':
    main()
