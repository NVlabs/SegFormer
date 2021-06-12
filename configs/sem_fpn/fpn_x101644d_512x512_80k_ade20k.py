_base_ = './fpn_r50_512x512_80k_ade20k.py'
model = dict(pretrained='open-mmlab://resnext101_64x4d',
             backbone=dict(
                 type='ResNeXt',
                 depth=101,
                 groups=64,
                 base_width=4))
