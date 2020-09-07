import os
import numpy
from PIL import Image
from torch.utils.data import Dataset

from typing import List


class iDataset(Dataset):
    '''A instance of Pytorch Dataloader.

    Attributes:
        image (array): The array of images.
        tf (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.

    Example:
        >>> rand_img = numpy.random.randint(0, 255, size=(1000, 32, 32, 3))
        >>> rand_img = numpy.uint8(rand_img)
        >>> ds = iDataset(rand_img)
        >>> print(ds[0])

    Raises:
        TypeError: Cannot handle this data type.

    '''
    def __init__(self, images, tf=None):
        '''Initialization of iDataset.

        Attributes:
            images: a numpy class store images, e.g. [32, 32, 3].
            transform: torchvision.transforms
        '''
        self.images = images
        self.tf = tf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = Image.fromarray(image)
        if self.tf is not None:
            image = self.transform(image)
        return image
