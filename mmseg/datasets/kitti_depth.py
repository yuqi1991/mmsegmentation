import os.path as osp
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from torch.utils.data import Dataset

from mmseg.core import mean_iou
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose
from .utils import read_calib_file, transform_from_rot_trans
import json


@DATASETS.register_module()
class KittiDepthDataset(Dataset):
    """Custom dataset for semantic segmentation.

    An example of file structure is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 depth_dir=None,
                 depth_suffix='.pcd',
                 pose_file=None,
                 cam_intrinc_file=None,
                 num_scales=4,
                 split="",
                 data_root="",
                 img_idx_file=None,
                 idx_file=None,
                 test_mode=False,
                 ignore_index=255,
                 ref_seq_id=None,
                 reduce_zero_label=False):

        if ref_seq_id is None:
            ref_seq_id = [-1, 1]

        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.depth_dir = depth_dir
        self.depth_suffix = depth_suffix
        self.num_scales = num_scales
        self.split = split
        self.data_root = data_root
        self.idx_file = idx_file
        self.img_idx_file = img_idx_file
        self.pose_file = pose_file
        self.cam_intrinc_file = cam_intrinc_file
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.ref_seq_id = ref_seq_id

        self.pose_cache = {}
        self.cam_cache = {}
        self.imu2cam_cache = {}

        if self.idx_file is not None and self.img_idx_file is not None:
            self.idx_file = osp.join(self.data_root, self.idx_file)
            self.img_idx_file = osp.join(self.data_root, self.img_idx_file)
        else:
            raise FileNotFoundError('Index File Not Defined')

        # read all img infos
        with open(self.img_idx_file, 'r') as f:
            self.img_infos = f.read().splitlines()
            self.img_infos = [json.loads(line) for line in self.img_infos]

        # read all idx infos
        with open(self.idx_file, 'r') as f:
            self.idx_file = f.read().splitlines()
            self.idx_file = [int(line) for line in self.idx_file]


    def __len__(self):
        """Total number of samples of data."""
        return len(self.idx_file)


    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def get_depth_info(self, idx):

        info = self.img_infos[idx]
        path = osp.join(self.data_root,info['split'], info['sequence'],
                        self.depth_dir, info['img']+self.depth_suffix)
        return dict(filename=path)

    def get_img_info(self, idx):
        info = self.img_infos[idx]
        path = osp.join(self.data_root,info['split'], info['sequence'],
                        self.img_dir, info['img']+self.img_suffix)
        return dict(filename=path)

    def get_pose(self, idx):
        info = self.img_infos[idx]
        path = osp.join(self.data_root,info['split'], info['sequence'],
                        self.pose_file)
        with open(path, 'r') as f:
            all_poses = f.read().splitlines()
        pose = all_poses[int(info['img'])].split(sep=' ')
        pose = np.asarray(pose).reshape(3,4)
        return pose

    def get_cam_K(self, idx):
        info = self.img_infos[idx]
        if info['split']+"_"+info['sequence'] not in self.imu2cam_cache:
            path = osp.join(self.data_root,info['split'], info['sequence'],
                            self.cam_intrinc_file)
            with open(path, 'r') as f:
                raw_cam_K = [line.split(sep=' ') for line in f.read().splitlines()]
                cam_K = []
                for line in raw_cam_K:
                    cam_K.extend(line)
                cam_K = np.asarray(cam_K).reshape(3,3)
            self.cam_cache[info['split']+"_"+info['sequence']] = cam_K
        else:
            cam_K = self.cam_cache[info['split']+"_"+info['sequence']]
        return cam_K

    def get_imu2cam(self, idx):
        info = self.img_infos[idx]
        if info['split'] not in self.imu2cam_cache:
            cam2cam = read_calib_file(osp.join(self.data_root, info['split'],
                                                'calib_cam_to_cam.txt'))
            imu2velo = read_calib_file(osp.join(self.data_root, info['split'],
                                                'calib_imu_to_velo.txt'))
            velo2cam = read_calib_file(osp.join(self.data_root, info['split'],
                                                'calib_velo_to_cam.txt'))
            velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])
            imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])
            cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))
            imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat
            self.imu2cam_cache[info['split']]=imu2cam
        else:
            imu2cam = self.imu2cam_cache[info['split']]
        return imu2cam


    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['depth_fields']= []
        return results

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        results = dict()

        seq_idx = self.idx_file[idx]
        img_info = self.get_img_info(seq_idx)
        ref_img_info = [self.get_img_info(seq_idx+ref_id) for ref_id in self.ref_seq_id]

        # ann_info = self.get_ann_info(idx)
        depth_info = self.get_depth_info(seq_idx)

        if self.pose_file:
            if seq_idx not in self.pose_cache:
                pose = self.get_pose(seq_idx)
                ref_pose = [self.get_pose(seq_idx+ref_id) for ref_id in self.ref_seq_id]
                self.pose_cache[seq_idx] = [pose].extend(ref_pose)
            else:
                pose = self.pose_cache[seq_idx][0]
                ref_pose = self.pose_cache[1:]
            results.update(dict(with_pose_gt=True,
                                pose=pose,
                                ref_pose=ref_pose,))
        else:
            results.update(dict(with_pose_gt=False))

        imu2cam = self.get_imu2cam(seq_idx)

        if self.cam_intrinc_file is not None:
            cam_K = self.get_cam_K(seq_idx)
            results.update(dict(cam_K=cam_K))

        results.update(dict(img_info=img_info,
                       # ann_info=ann_info,
                       ref_img_info=ref_img_info,
                       depth_info=depth_info,
                       depth_suffix=self.depth_suffix,
                       imu2cam=imu2cam,
                       ref_seq_id=self.ref_seq_id))

        results = self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """

        results = dict()

        seq_idx = self.idx_file[idx]
        img_info = self.get_img_info(seq_idx)
        results = dict(img_info=img_info)
        results = self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            gt_seg_map = mmcv.imread(
                img_info['ann']['seg_map'], flag='unchanged', backend='pillow')
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255

            gt_seg_maps.append(gt_seg_map)

        return gt_seg_maps

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mIoU']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        all_acc, acc, iou = mean_iou(
            results, gt_seg_maps, num_classes, ignore_index=self.ignore_index)
        summary_str = ''
        summary_str += 'per class results:\n'

        line_format = '{:<15} {:>10} {:>10}\n'
        summary_str += line_format.format('Class', 'IoU', 'Acc')
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        for i in range(num_classes):
            iou_str = '{:.2f}'.format(iou[i] * 100)
            acc_str = '{:.2f}'.format(acc[i] * 100)
            summary_str += line_format.format(class_names[i], iou_str, acc_str)
        summary_str += 'Summary:\n'
        line_format = '{:<15} {:>10} {:>10} {:>10}\n'
        summary_str += line_format.format('Scope', 'mIoU', 'mAcc', 'aAcc')

        iou_str = '{:.2f}'.format(np.nanmean(iou) * 100)
        acc_str = '{:.2f}'.format(np.nanmean(acc) * 100)
        all_acc_str = '{:.2f}'.format(all_acc * 100)
        summary_str += line_format.format('global', iou_str, acc_str,
                                          all_acc_str)
        print_log(summary_str, logger)

        eval_results['mIoU'] = np.nanmean(iou)
        eval_results['mAcc'] = np.nanmean(acc)
        eval_results['aAcc'] = all_acc

        return eval_results
