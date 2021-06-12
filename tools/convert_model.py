import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from IPython import embed
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('oldmodel', help='train config file path')
    parser.add_argument('newmodel', help='train config file path')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()
    model = torch.load(args.oldmodel)
    old_dict = model['state_dict']

    new_dict = OrderedDict()

    for old_key in old_dict.keys():
        if 'hybrid_embed' in old_key:
            new_key = old_key.replace('hybrid_embed', 'linear')
            print("{} -> {}".format(old_key, new_key))
        elif 'conv_seg' in old_key:
            new_key = old_key.replace('conv_seg', 'linear_pred')
            ncls = old_dict[old_key].shape[0]
            if  'bias' in old_key:
                rand_weight_bias = torch.randn(ncls)
                new_dict[old_key] = rand_weight_bias
            else:
                rand_weight_conv = torch.randn(ncls, 128, 1, 1)
                new_dict[old_key] = rand_weight_conv
            print("{} -> {}".format(old_key, new_key))
        else:
            new_key = old_key

        new_dict[new_key] = old_dict[old_key]


    model['state_dict'] = new_dict
    torch.save(model, args.newmodel)


if __name__ == "__main__":
    main()