import os
import glob
import shutil
import mmcv

def copy_lidar(load_dir, save_dir):
    # figure out the source directory
    lidar_source = os.path.join(load_dir, 'input', 'lidar')
    path_options = os.listdir(lidar_source)
    if len(path_options) != 1:
        raise ValueError(f'"{lidar_source}" contains {path_options} files. Expecting exactly 1 subdirectory.')
    lidar_source = os.path.join(lidar_source, path_options[0])
    # copy source files to processed directory to allow for easier downstream processing
    lidar_filenames = os.path.join(lidar_source, '*.pcd')
    destination = os.path.join(save_dir, 'inhouse_format', 'lidar')
    if not os.path.exists(destination):
        os.mkdir(destination)
    for _, filename in enumerate(mmcv.track_iter_progress(glob.glob(lidar_filenames))):
        shutil.copy(filename, destination)
