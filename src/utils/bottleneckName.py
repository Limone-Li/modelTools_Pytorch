import os
import sys
import abc

import torch
import torchvision
from torchvision import models

import numpy as np


def get_model_bottleneck_names(model, show_extra_repr=False):
    # We treat the extra repr like the sub-module, one item per line
    extra_lines = []

    # Extra_repr is the attribute of module. e.g 64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    extra_repr = model.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split('\n')

    child_lines = []
    for key, module in model._modules.items():
        mod_str = get_model_bottleneck_names(module)
        child_lines.extend([key + '.' + line for line in mod_str])

    lines = child_lines
    if show_extra_repr:
        lines.extend(extra_lines)

    if len(lines) == 0:
        main_name = model._get_name()
        return [main_name]

    return lines


def check_model_bottleneck_names(model, bottleneck_names):
    original_bottleneck_name = get_model_bottleneck_names(model, True)
    for name in bottleneck_names:
        answer = False
        for ori_name in original_bottleneck_name:
            if name in ori_name:
                answer = True
                break
        if answer is False:
            return False
    return True


model = models.vgg11()
name = get_model_bottleneck_names(model, False)
name[10] = 'haha'
print(check_model_bottleneck_names(model, name))
