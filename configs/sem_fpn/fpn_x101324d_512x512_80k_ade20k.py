_base_ = './fpn_r50_512x512_80k_ade20k.py'
model = dict(pretrained='open-mmlab://resnext101_32x4d',
             backbone=dict(
                 type='ResNeXt',
                 depth=101,
                 groups=32,
                 base_width=4))
