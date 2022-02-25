import argparse
import os
import numpy as np
import open3d
import glob
from scipy.spatial.transform import Rotation as R


LIDAR_DTYPE = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')]
LIDAR_EXT = [0, 0, 0, 0, 0, 0]
RADAR_EXT = [0, 0, 0, 0, 0, 0]

def load_lidar_pointcloud(file):
    points_structured = np.fromfile(file, dtype=LIDAR_DTYPE)
    points = np.column_stack((
        points_structured['x'],
        points_structured['y'],
        points_structured['z'],
    ))
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 1])
    return pcd  # todo: transform

def visualise(dir, ts):
    vis = open3d.visualization.VisualizerWithEditing()
    vis.create_window(width=1920,height=1080)
    lidar_file = os.path.join(dir, 'lidar', f'{ts}.bin')
    lidar_pc = load_lidar_pointcloud(lidar_file)
    lidar_pc.paint_uniform_color([0, 0, 1])
    # lidar_pc.transform(lidar_tr)
    vis.add_geometry(lidar_pc)
    # display
    # vis.poll_events()
    # vis.update_renderer()
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
        visualise(args.load_dir, ts)
        
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

if __name__ == '__main__':
    main()
