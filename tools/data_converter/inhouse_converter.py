import mmcv
import numpy as np
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R


class Inhouse2KITTI(object):
    """Inhouse to KITTI converter.

    This class serves as the converter to change the raw data to KITTI
    format.

    Args:
        load_dir (str): Directory to load inhouse raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (str): Number of workers for the parallel process.
        test_mode (bool): Whether in the test_mode. Default: False.
    """

    def __init__(self,
                 load_dir,
                 save_dir,
                 split,
                 workers=64,
                 test_mode=False):
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True

        self.inhouse_to_kitti_class_map = {
            0.0: 'DontCare',
            1.0: 'Pedestrian',
            2.0: 'Cyclist',
            3.0: 'Car',
            4.0: 'Truck'
        }

        # update extrinsic parameters
        self.lidar_transform = self.get_matrix_from_ext([0, 0, 0, -1, 2, 0])
        self.radar_transform = self.get_matrix_from_ext([0.06, -0.2, 0.2, -1, 2, 180])

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.split = split
        self.workers = int(workers)
        self.test_mode = test_mode

        assert split in {'training', 'validation', 'testing'}
        self.split_file = 'train.txt'
        if split == 'validation': self.split_file = 'val.txt'
        elif split == 'testing': self.split_file = 'test.txt'
        self.split_dir = 'testing' if self.test_mode else 'training'

        self.gt_path = os.path.join(self.load_dir, 'gt')
        self.radar_path = os.path.join(self.load_dir, 'radar')
        self.lidar_path = os.path.join(self.load_dir, 'lidar')
        self.label_save_dir = os.path.join(self.save_dir, 'label')
        self.calib_save_dir = os.path.join(self.save_dir, 'calib')
        self.lidar_save_dir = os.path.join(self.save_dir, 'lidar')
        self.radar_save_dir = os.path.join(self.save_dir, 'radar')
        for dir in [self.label_save_dir, self.calib_save_dir, self.lidar_save_dir, self.radar_save_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        self.timestamps = self._produce_timestamps()
        self.create_folder()

    def _produce_timestamps(self):
        split_path = os.path.join(self.save_dir, 'ImageSets', self.split_file)
        with open(split_path, 'r') as f:
            timestamps = [line.rstrip() for line in f]
        print(f'Loaded {len(timestamps)} timestamps from "{split_path}"')
        return timestamps

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        mmcv.track_parallel_progress(self.convert_one, self.timestamps, self.workers)
        print('\nFinished ...')

    def convert_one(self, ts):
        self.save_calib(ts)
        self.save_lidar(ts)
        self.save_radar(ts)
        if not self.test_mode:
            self.save_label(ts)

    def __len__(self):
        return len(self.timestamps)

    def save_radar(self, ts):
        """Convert the radar data from CSV to BIN format"""
        radar_load_path = os.path.join(self.radar_path, f'{ts}.csv')
        radar_save_path = os.path.join(self.radar_save_dir, f'{ts}.bin')
        try:
            radar_data = np.loadtxt(
                radar_load_path,
                delimiter=',',
                ndmin=1,
                skiprows=1,
                dtype=[
                    ('id', 'u4'),
                    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('vx', 'f4'), ('vy', 'f4'),
                    ('fPower', 'f4'), ('fRCS', 'f4'), ('fSpeed', 'f4')
                ]
            )
        except ValueError:
            print(f'Failed to convert radar for timestamp {ts}. Path: {radar_load_path}')
            raise
        # transform to ground-truth coordinate system
        radar_points_xyz = np.column_stack((radar_data['x'], radar_data['y'], radar_data['z']))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(radar_points_xyz)
        pcd.transform(self.radar_transform)
        # save transformed pointcloud
        radar_points_aux = np.column_stack((radar_data['fSpeed'], radar_data['fPower'], radar_data['fRCS']))
        radar_data = np.column_stack((pcd.points, radar_points_aux))
        radar_data.astype(np.float32).tofile(radar_save_path)

    def save_lidar(self, ts):
        """Convert the lidar data from PCD to BIN format"""
        pcd_file = os.path.join(self.lidar_path, f'{ts}.pcd')
        pcd_data = o3d.io.read_point_cloud(pcd_file)
        # transform to ground-truth coordinate system
        pcd_data.transform(self.lidar_transform)
        pc_path = os.path.join(self.lidar_save_dir, f'{ts}.bin')
        intensity = np.ones((len(pcd_data.points), ))
        point_cloud = np.column_stack((pcd_data.points, intensity))
        point_cloud.astype(np.float32).tofile(pc_path)

    def save_label(self, ts):
        """Parse and save the label data in txt format.
        The relation between inhouse and kitti coordinates is noteworthy:
        1. l,w,h (inhouse) --> h,w,l (kitti)
        """
        label_load_path = os.path.join(self.gt_path, f'{ts}.csv')
        label_save_path = os.path.join(self.label_save_dir, f'{ts}.txt')
        try: 
            labels = np.loadtxt(
                label_load_path,
                delimiter=' ',
                ndmin=1,
                dtype=[
                    ('id', 'u1'), ('class', 'u1'),
                    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('l', 'f4'), ('w', 'f4'), ('h', 'f4'),
                    ('rx', 'f4'), ('ry', 'f4'), ('rz', 'f4'),
                    ('unknown', 'u4')
                ]
            )
        except ValueError:
            print(f'Failed to convert labels for timestamp {ts}. Path: {label_load_path}')
            raise
        with open(label_save_path, 'w') as fp:
            for label in labels:
                fp.write(f'{self.inhouse_to_kitti_class_map[label[1]]} -1 -1 -10 -1 -1 -1 -1 '\
                    f'{label[7]:.2f} {label[6]:.2f} {label[5]:.2f} '\
                    f'{label[2]:.2f} {label[3]:.2f} {label[4]:.2f} '\
                    f'{label[9]:.2f}\n')

    def save_calib(self, ts):
        """Parse and save the calibration data."""
        R0_rect = [f'{i:e}' for i in np.eye(3).flatten()]
        calib_context = ''
        calib_context += 'R0_rect' + ': ' + ' '.join(R0_rect) + '\n'

        calib_save_path = os.path.join(self.calib_save_dir, f'{ts}.txt')
        with open(calib_save_path, 'w') as fp:
            fp.write(calib_context)
            fp.close()

    def create_folder(self):
        """Create folder for data preprocessing."""
        dir_list = [
            self.calib_save_dir,
            self.lidar_save_dir,
            self.radar_save_dir,
        ]
        for d in dir_list:
            mmcv.mkdir_or_exist(d)
        if not self.test_mode:
            mmcv.mkdir_or_exist(self.label_save_dir)

    def get_matrix_from_ext(self, ext):
        rot = R.from_euler('ZYX', ext[3:], degrees=True)
        rot_m = rot.as_matrix()
        x, y, z = ext[:3]
        tr = np.eye(4)
        tr[:3,:3] = rot_m
        tr[:3, 3] = np.array([x, y, z]).T
        return tr

    def cart_to_homo(self, mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.

        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.

        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret
