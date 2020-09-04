import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class ConceptDataset(Dataset):
    """A interface Dataloader dataset
    """
    def __init__(self, images, transform=None):
        """
        Args:
            images: a numpy class store images, [3, 32, 32]
            transform: torchvision.transforms
        """
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if len(np.shape(image)) > 2 and np.shape(image)[2] == 1:
            image = Image.fromarray(image[:, :, 0])
        elif len(np.shape(image)) > 2 and np.shape(image)[0] == 1:
            image = Image.fromarray(image[0, :, :])
        else:
            image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image