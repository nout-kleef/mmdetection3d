import argparse
import os
import numpy as np
import open3d


LIDAR_DTYPE = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')]

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

def visualise(vis, dir, timestamp, lidar=True, radar=False):
    lidar_file = os.path.join(dir, 'lidar', f'{timestamp}.bin')
    radar_file = os.path.join(dir, 'radar', f'{timestamp}.bin')
    if lidar:
        lidar_pc = load_lidar_pointcloud(lidar_file)
        vis.update_geometry(lidar_pc)
    vis.poll_events()
    vis.update_renderer()

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

    vis = open3d.visualization.Visualizer()
    # vis.create_window()
    for ts in args.timestamps:
        print(f'Visualising {ts}')
        visualise(vis, args.load_dir, ts)
        if save_img:
            img_name = os.path.join(args.save_dir, f'{ts}.png')
            vis.capture_screen_image(img_name)

if __name__ == '__main__':
    main()
