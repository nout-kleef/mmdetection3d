import argparse
import shutil
from os import path as osp
import read_raw, extract_gt, pcl_sync, vis_pcl

def main():
    parser = argparse.ArgumentParser(description='Read raw radar sweeps')
    parser.add_argument('data_root', help='path to root of directory containing unprocessed data')
    parser.add_argument('--delete_existing', action='store_true', help='if preprocessed data already exists, delete prior to new run')
    parser.add_argument('--skip_read_raw', action='store_true', help='skip read_raw.py')
    parser.add_argument('--skip_extract_gt', action='store_true', help='skip extract_gt.py')
    parser.add_argument('--skip_pcl_sync', action='store_true', help='skip pcl_sync.py')
    parser.add_argument('--skip_vis_pcl', action='store_true', help='skip vis_pcl.py')
    args = parser.parse_args()

    if args.delete_existing:
        print('*** STEP 0 - delete existing data ***')
        paths = [
            osp.join(args.data_root, 'sync_gt'),
            osp.join(args.data_root, 'sync_radar'),
            osp.join(args.data_root, 'radar_front'),
        ]
        for path in paths:
            if osp.exists(path):
                shutil.rmtree(path)
                print(f'deleted "{path}"')

    print('*** STEP 1 - read_raw.py ***')
    if not args.skip_read_raw: read_raw.read_raw(args.data_root)
    else: print('(skipped)')

    print('*** STEP 2 - extract_gt.py ***')
    if not args.skip_extract_gt: extract_gt.extract_gt(args.data_root)
    else: print('(skipped)')

    print('*** STEP 3 - pcl_sync.py ***')
    if not args.skip_pcl_sync: pcl_sync.pcl_sync(args.data_root)
    else: print('(skipped)')

    print('*** STEP 4 - vis_pcl.py ***')
    if not args.skip_vis_pcl: vis_pcl.vis_pcl(args.data_root)
    else: print('(skipped)')

    print('*** PREPROCESSING COMPLETED ***')

if __name__ == '__main__':
    main()
