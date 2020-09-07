import os
import sys
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import numpy as np
import unittest
from pathlib import Path

from src.modelTool import ModelTool


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B':
    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ],
}


class TestmodelTool(unittest.TestCase):
    def setUp(self):
        self.model = models.VGG(make_layers(cfgs['A']), 10)
        self.model_name = 'vgg'
        self.file_path = './tmp/vgg.pth'
        self.mt = ModelTool(self.model, self.model_name, self.file_path)

        self.train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        self.train_set = datasets.MNIST(
            './tmp/dataset/mnist',
            train=True,
            transform=self.train_transform,
            download=True,
        )
        self.test_set = datasets.MNIST('./tmp/dataset/mnist',
                                       train=False,
                                       transform=self.test_transform,
                                       download=True)

        self.train_loader = DataLoader(self.train_set, 128)
        self.test_loader = DataLoader(self.test_set, 128)

    def test_autotrain(self):
        if (Path(self.file_path).exists() is False):
            self.mt.auto_train(self.train_loader,
                               self.test_loader,
                               epoch_max=1,
                               verbose=True)

    def test_resume(self):
        self.mt.resume()
        print(self.mt)

    def test_save(self):
        if (Path(self.file_path).exists() is False):
            self.mt.save()

    def test_get_bottleneck_name(self):
        self.assertEqual(self.mt._bottleneck_name[0], '')
        self.assertEqual(self.mt._bottleneck_name[1], 'features')

    def test_run_examples(self):
        tf = transforms.Compose([transforms.ToTensor()])
        test_image = np.random.randint(0, 255, size=(10, 32, 32))
        test_image = np.uint8(test_image)
        ac, end, inp = self.mt.run_examples(test_image,
                                            transform=tf,
                                            verbose=False)


if __name__ == '__main__':
    unittest.main()
