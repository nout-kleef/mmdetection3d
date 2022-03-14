import os
import glob
import shutil
from pathlib import Path
import numpy as np
import mmcv
import torch

__seed = 3248723

def _create_random_subsets(elems, split):
    if not isinstance(split, list) or sum(split) != 1.0 or not all(val > 0.0 for val in split):
        raise ValueError(f'{split} must be a list whose values sum to 1.0, with no negative values.')
    assert isinstance(split, list)
    assert sum(split) == 1.0
    N = len(elems)
    # convert fractional split into respective number of elements
    counts = np.zeros((len(split), ), dtype=np.int32)
    for i in range(len(split) - 1):
        counts[i] = round(N * split[i])
    remaining = N - sum(counts)
    if remaining < 0: 
        raise ValueError(f'{remaining} elements left to distribute')
    counts[-1] = remaining
    print(f'INFO: split "{split}" resulted in the following subset sizes: {counts}')
    return torch.utils.data.random_split(elems, counts, generator=torch.Generator().manual_seed(__seed))

def _extract_values(subset):
    return [subset.dataset[i] for i in subset.indices]

def _get_timestamps(data_dir):
    gt_files = os.path.join(data_dir, 'inhouse_format', 'gt', '*.csv')
    return [Path(file).stem for file in sorted(glob.glob(gt_files))]

def _write_split_file(path, timestamps):
    with open(path, 'w') as f:
        f.writelines(ts + '\n' for ts in timestamps)

def partition(data_dir):
    split_files_dir = os.path.join(data_dir, 'kitti_format', 'ImageSets')
    timestamps = _get_timestamps(data_dir)
    subsets = _create_random_subsets(timestamps, [0.65, 0.15, 0.20])
    train, val, test = _extract_values(subsets[0]), _extract_values(subsets[1]), _extract_values(subsets[2])
    trainval = train + val
    assert len(set(train)) + len(set(val)) + len(set(test)) == len(timestamps)
    assert len(set(trainval)) == len(train) + len(val)
    if not os.path.exists(split_files_dir):
        os.makedirs(split_files_dir)
    _write_split_file(os.path.join(split_files_dir, 'train.txt'), train)
    _write_split_file(os.path.join(split_files_dir, 'val.txt'), val)
    _write_split_file(os.path.join(split_files_dir, 'trainval.txt'), trainval)
    _write_split_file(os.path.join(split_files_dir, 'test.txt'), test)
