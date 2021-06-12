_base_ = './fpn_r50_512x512_80k_ade20k.py'
model = dict(pretrained='open-mmlab://resnet18_v1c',
             backbone=dict(depth=18),
             neck=dict(in_channels=[64, 128, 256, 512]))
