# SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

This repository contains PyTorch evaluation code, training code and pretrained models for [SegFormer](https://arxiv.org/abs/2105.15203).

SegFormer is a simple, efficient and powerful semantic segmentation method, as shown in Figure 1.

We use [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0) as the codebase.

<!-- ![image](resources/image.png) -->
<div align="center">
  <img src="./resources/image.png" height="400">
</div>
<p align="center">
  Figure 1: Performance of SegFormer-B0 to SegFormer-B5.
</p>



## Install

For install and data preparation, please refer to the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).


## Evaluation

Download [trained weights](https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA?usp=sharing).

Example: evaluate ```SegFormer-B1``` on ```ADE20K```:

```
# single-gpu testing
python tools/test.py local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /path/to/checkpoint_file

# multi-gpu testing
./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /path/to/checkpoint_file <GPU_NUM>

# multi-gpu, multi-scale testing
tools/dist_test.sh local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /path/to/checkpoint_file <GPU_NUM> --aug-test
```

## Training

Download [pre-trained weights](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing) pretrained on ImageNet-1K, and put them in a folder ```pretrained/```.

Example: train ```SegFormer-B1``` on ```ADE20K```:

```
# single-gpu training
python tools/train.py local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py 

# multi-gpu training
./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py <GPU_NUM>
```

## License
Please check the LICENSE file. SegFormer may be used non-commercially, meaning for research or 
evaluation purposes only. For business inquiries, please contact 
[researchinquiries@nvidia.com](mailto:researchinquiries@nvidia.com).


## Citing SegFormer
```
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={arXiv preprint arXiv:2105.15203},
  year={2021}
}
```
