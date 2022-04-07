import argparse
import os
import numpy as np
import open3d
from scipy.spatial.transform import Rotation as R
from line_mesh import LineMesh


# LIDAR_DTYPE = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')]
RADAR_DTYPE = [
    ('vec.x', 'f4'), ('vec.y', 'f4'), ('vec.z', 'f4'), ('vel.x', 'f4'), ('vel.y', 'f4'), ('vel.z', 'f4'), ('detect.velocity', 'f4')
]
GT_DTYPE = dtype=[
    ('class', 'U8'),
    ('truncated', 'i1'), ('occluded', 'i1'), ('alpha', 'i1'),
    ('bbox0', 'i1'), ('bbox1', 'i1'), ('bbox2', 'i1'), ('bbox3', 'i1'),
    ('h', 'f4'), ('w', 'f4'), ('l', 'f4'),
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('yaw', 'f4')
]
VSCALE = 0.07

def get_matrix_from_ext(ext):
    rot = R.from_euler('ZYX', ext[3:], degrees=True)
    rot_m = rot.as_matrix()
    x, y, z = ext[:3]
    tr = np.eye(4)
    tr[:3,:3] = rot_m
    tr[:3, 3] = np.array([x, y, z]).T
    return tr


def create_lines(points, vs):
    def produce_line_extension(p, v):
        """Produce _p, a point along the line through O and p, with distance v * VSCALE from p"""
        direction = p / np.sum(p)
        dist = v * VSCALE
        return p + direction * dist

    N = len(points)
    aux_points = np.zeros_like(points)
    for i, p in enumerate(points):
        v = vs[i]
        aux_points[i] = produce_line_extension(p, v)
    points_aug = np.concatenate((points, aux_points))
    conns = np.column_stack((np.arange(0, N), np.arange(N, N + N)))
    return points_aug, open3d.utility.Vector2iVector(conns)

def load_pointcloud(file, dtype, color, velocity=None):
    points_structured = np.loadtxt(file, dtype=dtype, delimiter=',', ndmin=1)
    points = np.column_stack((
        points_structured['vec.x'],
        points_structured['vec.y'],
        points_structured['vec.z'],
    ))
    if velocity is None:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)
        return pcd
    vs = points_structured[velocity]
    points_aug, connections = create_lines(points, vs)
    lines = LineMesh(points_aug, connections, colors=color, radius=0.1)
    return lines

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

def visualise(args, frame):
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=1280,height=720)
    # lidar
    # lidar_file = os.path.join(args.load_dir, 'lidar', f'{ts}.bin')
    # lidar_pc = load_pointcloud(lidar_file, LIDAR_DTYPE, lidar_transform, [0, 0, 1])
    # radar
    radar_file = os.path.join(args.load_dir, 'pcl', f'pcl_{frame}.csv')
    radar_pc = load_pointcloud(radar_file, RADAR_DTYPE, [1, 0, 0], velocity='detect.velocity')
    # gt
    # gt_file = os.path.join(args.load_dir, 'label', f'{ts}.txt')
    # boxes = load_gt(gt_file)
    # display
    # vis.add_geometry(lidar_pc)
    for cyl_seg in radar_pc.cylinder_segments:
        vis.add_geometry(cyl_seg)
    # for box in boxes:
    #     vis.add_geometry(box)
    vis.run()

def main():
    parser = argparse.ArgumentParser(description='Visualise lidar + radar pointclouds')
    parser.add_argument('--load_dir', help='Directory containing data')
    parser.add_argument('--save_dir', help='Directory used to save images (optional)')
    parser.add_argument('frames', nargs='*')
    args = parser.parse_args()

    save_img = args.save_dir is not None
    if save_img:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        else:
            os.system("rm -r " + args.save_dir)
            os.mkdir(args.save_dir)
    
    for frame in args.frames:
        visualise(args, frame)

def get_rotation(yaw):
    angle = np.array([0, 0, yaw])
    r = R.from_euler('XYZ', angle)
    return r.as_matrix()

if __name__ == '__main__':
    main()
