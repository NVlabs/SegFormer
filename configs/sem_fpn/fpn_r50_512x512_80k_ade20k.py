_base_ = [
    '../_base_/models/fpn_r50.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py'
]
model = dict(decode_head=dict(num_classes=150))

gpu_factor = 2 #mmseg默认4卡训练 我这边8卡的话 lr*2, iter/2
# optimizer
optimizer = dict(type='SGD', lr=0.01*gpu_factor, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000//gpu_factor)
checkpoint_config = dict(by_epoch=False, interval=8000//gpu_factor)
evaluation = dict(interval=8000, metric='mIoU')

