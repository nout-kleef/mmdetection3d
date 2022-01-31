import argparse
import shutil
import os
import read_raw, extract_gt, pcl_sync, copy_lidar, vis_pcl

def _purge(paths):
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f'deleted "{path}"')

def main():
    parser = argparse.ArgumentParser(description='Read raw radar sweeps')
    parser.add_argument('load_dir', help='path to root of directory containing unprocessed data')
    parser.add_argument('save_dir', help='path to root of directory to store processed data')
    parser.add_argument('--delete_existing', action='store_true', help='if preprocessed data already exists, delete prior to new run')
    parser.add_argument('--skip_read_raw', action='store_true', help='skip read_raw.py')
    parser.add_argument('--skip_extract_gt', action='store_true', help='skip extract_gt.py')
    parser.add_argument('--skip_pcl_sync', action='store_true', help='skip pcl_sync.py')
    parser.add_argument('--skip_copy_lidar', action='store_true', help='skip copy_lidar.py')
    parser.add_argument('--skip_vis_pcl', action='store_true', help='skip vis_pcl.py')
    args = parser.parse_args()

    _raw_paths = [
        os.path.join(args.save_dir, 'inhouse_format', 'radar_raw'),
        os.path.join(args.save_dir, 'inhouse_format', 'gt_raw'),
    ]
    _processed_paths = [
        os.path.join(args.save_dir, 'inhouse_format', 'gt'),
        os.path.join(args.save_dir, 'inhouse_format', 'radar'),
        os.path.join(args.save_dir, 'inhouse_format', 'lidar'),
    ]

    if args.delete_existing:
        print('*** STEP 0 - delete existing data ***')
        _purge(_raw_paths + _processed_paths)

    print('*** STEP 1 - read_raw.py ***')
    if not args.skip_read_raw: read_raw.read_raw(args.load_dir, args.save_dir)
    else: print('(skipped)')

    print('*** STEP 2 - extract_gt.py ***')
    if not args.skip_extract_gt: extract_gt.extract_gt(args.load_dir, args.save_dir)
    else: print('(skipped)')

    print('*** STEP 3 - pcl_sync.py ***')
    if not args.skip_pcl_sync:
        pcl_sync.pcl_sync(args.load_dir, args.save_dir)
        _purge(_raw_paths)  # remove unsynced gt and radar data to save storage
    else: print('(skipped)')

    print('*** STEP 4 - copy_lidar.py ***')
    if not args.skip_copy_lidar: copy_lidar.copy_lidar(args.load_dir, args.save_dir)
    else: print('(skipped)')

    print('*** STEP 5 - vis_pcl.py ***')
    if not args.skip_vis_pcl: vis_pcl.vis_pcl(args.load_dir, args.save_dir)
    else: print('(skipped)')

    print('*** PREPROCESSING COMPLETED ***')

if __name__ == '__main__':
    main()
