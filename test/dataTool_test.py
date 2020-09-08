import os
import sys
import unittest
import numpy
import torchvision
from torchvision import transforms

from tools.dataTool import iDataset


class TestiDataset(unittest.TestCase):
    def test_iDataset(self):
        self.rand_img = numpy.random.randint(0, 255, size=(100, 32, 32, 3))
        self.rand_img = numpy.uint8(self.rand_img)
        ds = iDataset(self.rand_img)
        self.assertTrue((self.rand_img[0] == numpy.array(ds[0])).all())


if __name__ == '__main__':
    unittest.main()
