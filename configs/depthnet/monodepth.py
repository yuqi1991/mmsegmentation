_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/kitti_depth.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    pretrained=None,
    decode_head=dict(num_classes=21), 
    auxiliary_head=dict(num_classes=21))
