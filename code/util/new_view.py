
import numpy as np
from skimage import color
#scikit-image
import torch
import torchvision.datasets as datasets
from torchvision import transforms

mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]

def img_to_pil(img):
    normalize = transforms.Normalize(mean=mean, std=std)
    loader = transforms.Compose([transforms.ToTensor(),normalize])
    img = loader(img)
    return img

def RGB2Lab(img):
    img = np.asarray(img, np.uint8)
    img = color.rgb2lab(img)
    return img

'''
class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        return img

'''
class RGB2HSV(object):
    """Convert RGB PIL image to ndarray HSV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2hsv(img)
        return img


class RGB2HED(object):
    """Convert RGB PIL image to ndarray HED."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2hed(img)
        return img


class RGB2LUV(object):
    """Convert RGB PIL image to ndarray LUV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img =color.rgb2luv(img)
        return img


class RGB2YUV(object):
    """Convert RGB PIL image to ndarray YUV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yuv(img)
        return img


class RGB2XYZ(object):
    """Convert RGB PIL image to ndarray XYZ."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2xyz(img)
        return img


class RGB2YCbCr(object):
    """Convert RGB PIL image to ndarray YCbCr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ycbcr(img)
        return img


class RGB2YDbDr(object):
    """Convert RGB PIL image to ndarray YDbDr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ydbdr(img)
        return img


class RGB2YPbPr(object):
    """Convert RGB PIL image to ndarray YPbPr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ypbpr(img)
        return img


class RGB2YIQ(object):
    """Convert RGB PIL image to ndarray YIQ."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yiq(img)
        return img


class RGB2CIERGB(object):
    """Convert RGB PIL image to ndarray RGBCIE."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2rgbcie(img)
        return img
