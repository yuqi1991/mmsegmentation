import argparse, os
import os.path as osp

import mmcv

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Kitti raw dataset index file')
    parser.add_argument('--root_path', help='cityscapes data path', default="data/kitti_raw")
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args

def arrange_data(root_dir):
    resualt = list()


    for dirpath, dirnames, filenames in os.walk(root_dir):
        splits = dirnames

    return resualt

if __name__ == '__main__':
    args = parse_args()
    cityscapes_path = args.root_path
    result = arrange_data(args.root_path)
    print("done")