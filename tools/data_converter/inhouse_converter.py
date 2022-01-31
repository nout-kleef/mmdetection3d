import mmcv
import numpy as np
from pathlib import Path
from glob import glob
import os
from os.path import join
import open3d as o3d


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
                 prefix,
                 workers=64,
                 test_mode=False):
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True

        self.selected_inhouse_classes = [
            # 0.0,  # unknown
            1.0,  # pedestrian
            2.0,  # cyclist
            3.0,  # small vehicle
            4.0   # large vehicle
        ]

        # Only data collected in specific locations will be converted
        # If set None, this filter is disabled
        # Available options: location_sf (main dataset)
        self.selected_inhouse_locations = None  # todo
        self.save_track_id = False
        
        # self.lidar_list = [
        #     '_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT',
        #     '_SIDE_LEFT'
        # ]
        # self.type_list = [
        #     'UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST'
        # ]
        self.inhouse_to_kitti_class_map = {
            '0.0': 'DontCare',
            '1.0': 'Pedestrian',
            '2.0': 'Cyclist',
            '3.0': 'Car',
            '4.0': 'Truck'
        }

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.prefix = prefix
        self.workers = int(workers)
        self.test_mode = test_mode

        self.gt_path, self.radar_path, self.lidar_path = self._produce_paths()

        # self.tfrecord_pathnames = sorted(
        #     glob(join(self.load_dir, '*.tfrecord')))
        self.timestamps = self._produce_timestamps()
        print(self.timestamps[0])

        self.label_save_dir = f'{self.save_dir}/label_'
        self.label_all_save_dir = f'{self.save_dir}/label_all'
        # self.image_save_dir = f'{self.save_dir}/image_'
        self.calib_save_dir = f'{self.save_dir}/calib'
        self.lidar_save_dir = f'{self.save_dir}/lidar'
        self.radar_save_dir = f'{self.save_dir}/radar'
        self.pose_save_dir = f'{self.save_dir}/pose'

        self.create_folder()

    def _produce_paths(self):
        gt_path = join(self.load_dir, 'sync_gt')
        radar_path = join(self.load_dir, 'sync_radar')
        # finding the LiDAR path is more complicated due to upstream file structure
        lidar_path = join(self.load_dir, 'input', 'lidar')
        path_options = os.listdir(lidar_path)
        if len(path_options) != 1:
            raise ValueError(f'"{lidar_path}" contains {path_options} files. Expecting exactly 1 subdirectory.')
        lidar_path = join(lidar_path, path_options[0])
        return gt_path, radar_path, lidar_path

    def _produce_timestamps(self):
        gt_files = sorted([Path(p) for p in glob(join(self.gt_path, '*.csv'))])
        radar_files = sorted([Path(p) for p in glob(join(self.radar_path, '*.csv'))])
        lidar_files = sorted([Path(p) for p in glob(join(self.lidar_path, '*.pcd'))])
        # perform checks to ensure files can be properly matched downstream
        num_files = len(gt_files)
        if num_files != len(radar_files) or num_files != len(lidar_files):
            raise ValueError(f'Mismatch in ground truth ({len(gt_files)} files), '
            f'radar ({len(radar_files)} files) and LiDAR ({len(lidar_files)} files) data')
        for i in range(num_files):
            ts = gt_files[i].stem
            if ts != radar_files[i].stem or ts != lidar_files[i].stem:
                raise ValueError(f'Mismatch in ground truth ({ts}), radar ({radar_files[i].stem}) '
                f'and LiDAR ({lidar_files[i].stem}) timestamps')
        return gt_files

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        mmcv.track_parallel_progress(self.convert_one, range(len(self)),
                                     self.workers)
        print('\nFinished ...')

    def convert_one(self, file_idx):
        """Convert action for single file.

        Args:
            file_idx (int): Index of the file to be converted.
        """
        pathname = self.tfrecord_pathnames[file_idx]
        # dataset = tf.data.TFRecordDataset(pathname, compression_type='')

        # for frame_idx, data in enumerate(dataset):

        #     frame = dataset_pb2.Frame()
        #     frame.ParseFromString(bytearray(data.numpy()))
        #     if (self.selected_inhouse_locations is not None
        #             and frame.context.stats.location
        #             not in self.selected_inhouse_locations):
        #         continue

        #     self.save_calib(frame, file_idx, frame_idx)
        #     self.save_lidar(frame, file_idx, frame_idx)
        #     self.save_pose(frame, file_idx, frame_idx)

        #     if not self.test_mode:
        #         self.save_label(frame, file_idx, frame_idx)

    def __len__(self):
        """Length of the filename list."""
        return len(self.timestamps)

    def save_calib(self, frame, file_idx, frame_idx):
        """Parse and save the calibration data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        # inhouse front camera to kitti reference camera
        T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                                       [1.0, 0.0, 0.0]])
        camera_calibs = []
        R0_rect = [f'{i:e}' for i in np.eye(3).flatten()]
        Tr_velo_to_cams = []
        calib_context = ''

        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
                4, 4)
            T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
            Tr_velo_to_cam = \
                self.cart_to_homo(T_front_cam_to_ref) @ T_vehicle_to_cam
            if camera.name == 1:  # FRONT = 1, see dataset.proto for details
                self.T_velo_to_front_cam = Tr_velo_to_cam.copy()
            Tr_velo_to_cam = Tr_velo_to_cam[:3, :].reshape((12, ))
            Tr_velo_to_cams.append([f'{i:e}' for i in Tr_velo_to_cam])

            # intrinsic parameters
            camera_calib = np.zeros((3, 4))
            camera_calib[0, 0] = camera.intrinsic[0]
            camera_calib[1, 1] = camera.intrinsic[1]
            camera_calib[0, 2] = camera.intrinsic[2]
            camera_calib[1, 2] = camera.intrinsic[3]
            camera_calib[2, 2] = 1
            camera_calib = list(camera_calib.reshape(12))
            camera_calib = [f'{i:e}' for i in camera_calib]
            camera_calibs.append(camera_calib)

        # all camera ids are saved as id-1 in the result because
        # camera 0 is unknown in the proto
        for i in range(5):
            calib_context += 'P' + str(i) + ': ' + \
                ' '.join(camera_calibs[i]) + '\n'
        calib_context += 'R0_rect' + ': ' + ' '.join(R0_rect) + '\n'
        for i in range(5):
            calib_context += 'Tr_velo_to_cam_' + str(i) + ': ' + \
                ' '.join(Tr_velo_to_cams[i]) + '\n'

        with open(
                f'{self.calib_save_dir}/{self.prefix}' +
                f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt',
                'w+') as fp_calib:
            fp_calib.write(calib_context)
            fp_calib.close()

    def save_lidar(self, frame):
        """Parse and save the lidar data in psd format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """

        # get PCD data
        # pcd_file = Path(self.lidar_path) / ts / '.pcd'
        # pcd_data = o3d.io.read_point_cloud(pcd_file)

        # # concatenate x,y,z, intensity, elongation, timestamp (6-dim)
        # point_cloud = np.column_stack(
        #     (points, intensity, elongation, timestamp))

        # pc_path = f'{self.lidar_save_dir}/{self.prefix}' + \
        #     f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.bin'
        # point_cloud.astype(np.float32).tofile(pc_path)

    def save_label(self, frame, file_idx, frame_idx):
        """Parse and save the label data in txt format.
        The relation between inhouse and kitti coordinates is noteworthy:
        1. x, y, z correspond to l, w, h (inhouse) -> l, h, w (kitti)
        2. x-y-z: front-left-up (inhouse) -> right-down-front(kitti)
        3. bbox origin at volumetric center (inhouse) -> bottom center (kitti)
        4. rotation: +x around y-axis (kitti) -> +x around z-axis (inhouse)

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        fp_label_all = open(
            f'{self.label_all_save_dir}/{self.prefix}' +
            f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt', 'w+')
        id_to_bbox = dict()
        id_to_name = dict()
        for labels in frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                # TODO: need a workaround as bbox may not belong to front cam
                bbox = [
                    label.box.center_x - label.box.length / 2,
                    label.box.center_y - label.box.width / 2,
                    label.box.center_x + label.box.length / 2,
                    label.box.center_y + label.box.width / 2
                ]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name - 1

        for obj in frame.laser_labels:
            bounding_box = None
            name = None
            id = obj.id
            for lidar in self.lidar_list:
                if id + lidar in id_to_bbox:
                    bounding_box = id_to_bbox.get(id + lidar)
                    name = str(id_to_name.get(id + lidar))
                    break

            if bounding_box is None or name is None:
                name = '0'
                bounding_box = (0, 0, 0, 0)

            my_type = self.type_list[obj.type]

            if my_type not in self.selected_inhouse_classes:
                continue

            if self.filter_empty_3dboxes and obj.num_lidar_points_in_box < 1:
                continue

            my_type = self.inhouse_to_kitti_class_map[my_type]

            height = obj.box.height
            width = obj.box.width
            length = obj.box.length

            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z - height / 2

            # project bounding box to the virtual reference frame
            pt_ref = self.T_velo_to_front_cam @ \
                np.array([x, y, z, 1]).reshape((4, 1))
            x, y, z, _ = pt_ref.flatten().tolist()

            rotation_y = -obj.box.heading - np.pi / 2
            track_id = obj.id

            # not available
            truncated = 0
            occluded = 0
            alpha = -10

            line = my_type + \
                ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                    round(truncated, 2), occluded, round(alpha, 2),
                    round(bounding_box[0], 2), round(bounding_box[1], 2),
                    round(bounding_box[2], 2), round(bounding_box[3], 2),
                    round(height, 2), round(width, 2), round(length, 2),
                    round(x, 2), round(y, 2), round(z, 2),
                    round(rotation_y, 2))

            if self.save_track_id:
                line_all = line[:-1] + ' ' + name + ' ' + track_id + '\n'
            else:
                line_all = line[:-1] + ' ' + name + '\n'

            fp_label = open(
                f'{self.label_save_dir}{name}/{self.prefix}' +
                f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt', 'a')
            fp_label.write(line)
            fp_label.close()

            fp_label_all.write(line_all)

        fp_label_all.close()

    def save_pose(self, frame, file_idx, frame_idx):
        """Parse and save the pose data.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(
            join(f'{self.pose_save_dir}/{self.prefix}' +
                 f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'),
            pose)

    def create_folder(self):
        """Create folder for data preprocessing."""
        if not self.test_mode:
            dir_list1 = [
                self.label_all_save_dir,
                self.calib_save_dir,
                self.lidar_save_dir,
                self.radar_save_dir,
                self.pose_save_dir,
            ]
            dir_list2 = [
                self.label_save_dir,
            ]
        else:
            dir_list1 = [
                self.calib_save_dir,
                self.lidar_save_dir,
                self.radar_save_dir,
                self.pose_save_dir
            ]
            dir_list2 = []
        for d in dir_list1:
            mmcv.mkdir_or_exist(d)
        for d in dir_list2:
            for i in range(5):
                mmcv.mkdir_or_exist(f'{d}{str(i)}') # todo

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
