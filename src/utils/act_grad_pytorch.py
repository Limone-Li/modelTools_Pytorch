import os
import six
import sys
from copy import deepcopy

from abc import ABCMeta
from abc import abstractmethod

import torch
from torch.autograd import grad
from torch.utils.data import DataLoader

import numpy as np


class ActivationAndGradientInterface(six.with_metaclass(ABCMeta, object)):
    """Interface for generating and getting activation and gradient"""
    @abstractmethod
    def process_activation(self, input_data, bottleneck_names):
        pass

    @abstractmethod
    def process_gradient(self, input_data, bottleneck_names, target):
        pass


class ActivationAndGradientBase(ActivationAndGradientInterface):
    """Base abstract class for progressing activation and gradient"""
    def __init__(self,
                 model,
                 input_data,
                 file_dir,
                 bottleneck_names,
                 target,
                 batch_size=1):
        self.model = model
        self.input_data = input_data
        self.file_dir = file_dir
        self.bottleneck_names = bottleneck_names
        self.target = target

    def get_model(self):
        return self.model

    def process_activation(self, input_data, bottleneck_names):
        acts = {}
        if not os.path.exists(self.file_dir):
            os.mkdir(self.file_dir)

    def get_activation_gradient(self, sub_input_data, verbose=False):
        """ Generates the concept activations and gradients"""
        target = self.target
        model = self.model

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
