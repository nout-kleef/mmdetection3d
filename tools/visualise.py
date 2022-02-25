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

def visualise(vis, lidar, curr_lidar_pc=None, curr_radar_pc=None):
    lidar_pc = load_lidar_pointcloud(lidar)
    # lidar_pc.transform(lidar_tr)
    if curr_lidar_pc:
        curr_lidar_pc.points = lidar_pc.points
        curr_lidar_pc.paint_uniform_color([0, 0, 1])
        vis.update_geometry(curr_lidar_pc)
    else:
        vis.add_geometry(lidar_pc)
        curr_lidar_pc = lidar_pc
    return curr_lidar_pc, None

def main():
    parser = argparse.ArgumentParser(description='Visualise lidar + radar pointclouds')
    parser.add_argument('--load_dir', help='Directory containing data')
    parser.add_argument('--save_dir', help='Directory used to save images (optional)')
    parser.add_argument('timestamps', nargs='*')
    args = parser.parse_args()
    
    print(args.load_dir)
    print(args.save_dir)
    print(args.timestamps)
    save_img = args.save_dir is not None

    lidar_files = sorted(glob.glob(os.path.join(args.load_dir, "lidar", "*.bin")))
    # gt_files = sorted(glob.glob(os.path.join(load_dir, "gt", "*.csv")))
    radar_files = sorted(glob.glob(os.path.join(args.load_dir, "radar", "*.bin")))
    if save_img:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        else:
            os.system("rm -r " + args.save_dir)
            os.mkdir(args.save_dir)

    start = 30
    # gt_fname = gt_files[4]
    lidar_tr = get_matrix_from_ext(LIDAR_EXT)
    radar_tr = get_matrix_from_ext(RADAR_EXT)
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=1920,height=1080)

    lidar_pc, radar_pc = visualise(vis, lidar_files[0], radar_files[0])

    # print(gt_fname)
    # gt_data = np.loadtxt(gt_fname)
    # box_list = []
    # for obj_info in gt_data:
    #     obj_bbx = get_bbx_param(obj_info)
    #     box_list += [obj_bbx]
    #     vis.add_geometry(obj_bbx)

    for idx in range(start, len(lidar_files)):
        lidar_pc, radar_pc = visualise(vis, lidar_files[idx], lidar_pc, radar_pc)
        # for box in box_list:
        #     vis.remove_geometry(box, reset_bounding_box=False)
        # box_list = []
        # gt_data = np.loadtxt(gt_files[idx])
        # for obj_info in gt_data:
        #     obj_bbx = get_bbx_param(obj_info)
        #     box_list += [obj_bbx]
        #     vis.add_geometry(obj_bbx, reset_bounding_box=False)
        vis.poll_events()
        vis.update_renderer()
        if save_img:
            fname = os.path.join(args.save_dir, f'{idx}.png')
            vis.capture_screen_image(fname)

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
