# SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

We use [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0) as the codebase.


## How to install

Install according to the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).


## Data preparation

Prepare ADE20K, Cityscapes according to the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).

## Evaluation

First, download trained weights from [google drive](https://drive.google.com/file/d/1AbNMxJYzP_JT1BJNtMc2M4REhH1tMZw7/view?usp=sharing). Here we provide weights of SegFormer-B1 on ADE20K.

For example, to evaluate SegFormer-B1 on ADE20K on a single node with 8 gpus run:

```
./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /path/to/checkpoint_file 8
```

