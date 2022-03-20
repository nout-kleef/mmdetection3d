import mmcv
import numpy as np
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from pyntcloud import PyntCloud


def get_matrix_from_ext(ext):
    rot = R.from_euler('ZYX', ext[3:], degrees=True)
    rot_m = rot.as_matrix()
    x, y, z = ext[:3]
    tr = np.eye(4)
    tr[:3,:3] = rot_m
    tr[:3, 3] = np.array([x, y, z]).T
    return tr

def get_date_key(ts):
    return '0118' if ts < 1643000000000 else '0126'
    
ext_params = {
    '0118': {
        'lidar': get_matrix_from_ext([0.00, 0.0, -0.3, -2.5, 0.0, 0]),
        'radar': get_matrix_from_ext([0.06, -0.2, 0.7, -3.5, 2.0, 180]),
    },
    '0126': {
        'lidar': get_matrix_from_ext([0.00, 0.0, 0.0, -1.0, 2.0, 0]),
        'radar': get_matrix_from_ext([0.06, -0.2, 0.2, -1.0, 2.0, 180]),
    }
}


class InhouseLabel2Kitti:
    def __init__(self, inhouse_label, classname_dict):
        self._class = classname_dict[inhouse_label[1]]
        self._trunc = -1
        self._occl = -1
        self._alpha = -10
        self._bbox = [-1, -1, -1, -1]
        self._length = inhouse_label[5]
        self._width = inhouse_label[6]
        self._height = inhouse_label[7]
        self._x = inhouse_label[2]
        self._y = inhouse_label[3]
        self._z = inhouse_label[4] - self._height / 2.0
        self._roty = inhouse_label[9]

    def __repr__(self) -> str:
        return f'{self._class} {self._trunc} {self._occl} {self._alpha} '\
            f'{self._bbox[0]} {self._bbox[1]} {self._bbox[2]} {self._bbox[3]} '\
            f'{self._height:.2f} {self._width:.2f} {self._length:.2f} '\
            f'{self._x:.2f} {self._y:.2f} {self._z:.2f} '\
            f'{self._roty:.2f}'

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

        self.radar_gt_path = os.path.join(self.load_dir, 'radar_gt')
        self.gt_path = os.path.join(self.load_dir, 'gt')
        self.radar_path = os.path.join(self.load_dir, 'radar')
        self.lidar_path = os.path.join(self.load_dir, 'lidar')
        self.label_save_dir = os.path.join(self.save_dir, 'label')
        self.calib_save_dir = os.path.join(self.save_dir, 'calib')
        self.lidar_save_dir = os.path.join(self.save_dir, 'lidar')
        self.radar_save_dir = os.path.join(self.save_dir, 'radar')
        self.radar_gt_save_dir = os.path.join(self.save_dir, 'label_r')
        for dir in [self.label_save_dir, self.calib_save_dir, self.lidar_save_dir, self.radar_save_dir, self.radar_gt_save_dir]:
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
        self.save_radar_label(ts)  # bboxes without nearby radar points filtered out
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
        collection_date = get_date_key(int(ts))
        pcd.transform(ext_params[collection_date]['radar'])
        # save transformed pointcloud
        radar_points_aux = np.column_stack((radar_data['fSpeed'], radar_data['fPower'], radar_data['fRCS']))
        radar_data = np.column_stack((pcd.points, radar_points_aux))
        radar_data.astype(np.float32).tofile(radar_save_path)

    def save_lidar(self, ts):
        """Convert the lidar data from PCD to BIN format"""
        pcd_file = os.path.join(self.lidar_path, f'{ts}.pcd')
        pcd_data = o3d.io.read_point_cloud(pcd_file)
        # transform to ground-truth coordinate system
        collection_date = get_date_key(int(ts))
        pcd_data.transform(ext_params[collection_date]['lidar'])
        pc_path = os.path.join(self.lidar_save_dir, f'{ts}.bin')
        intensity = self._get_intensity(pcd_file)
        point_cloud = np.column_stack((pcd_data.points, intensity))
        point_cloud.astype(np.float32).tofile(pc_path)

    def save_label(self, ts):
        """Parse and save the label data in txt format.
        The relation between inhouse and kitti coordinates is noteworthy:
        1. l,w,h (inhouse) --> h,w,l (kitti)
        2. bbox origin at volumetric center (inhouse) -> bottom center (kitti)
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
                kitti_label = InhouseLabel2Kitti(label, self.inhouse_to_kitti_class_map)
                fp.write(f'{kitti_label}\n')

    def save_radar_label(self, ts):
        """Parse and save the label data in txt format.
        The relation between inhouse and kitti coordinates is noteworthy:
        1. l,w,h (inhouse) --> h,w,l (kitti)
        """
        label_load_path = os.path.join(self.radar_gt_path, f'{ts}.csv')
        fsize = os.path.getsize(label_load_path)
        if fsize == 0:
            # skip processing empty file
            # self.radar_gt_path, f'{ts}.csv'
            save_path = os.path.join(self.radar_gt_save_dir, f'{ts}.txt')
            gt = np.array([])
            np.savetxt(save_path, gt, fmt='%s')
            pass
        else:
            
            label_save_path = os.path.join(self.radar_gt_save_dir, f'{ts}.txt')
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
            self.label_save_dir,
            self.radar_gt_save_dir
        ]
        for d in dir_list:
            mmcv.mkdir_or_exist(d)

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

    def _get_intensity(self, path) -> np.ndarray:
        pc = PyntCloud.from_file(path)
        return np.array(pc.points['intensity'])