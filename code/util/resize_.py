import numpy as np
import math
import cv2


def random_crop(img, scale=[0.8, 1.0], ratio=[3. / 4., 4. / 3.], resize_w=224, resize_h=224):
            """
            随机裁剪
            :param img:
            :param scale: 缩放
            :param ratio:
            :param resize_w:
            :param resize_h:
            :return:
            """
            aspect_ratio = math.sqrt(np.random.uniform(*ratio))
            w = 1. * aspect_ratio
            h = 1. / aspect_ratio
            src_h, src_w = img.shape[:2]
 
            bound = min((float(src_w) / src_h) / (w ** 2),
                        (float(src_h) / src_w) / (h ** 2))
            scale_max = min(scale[1], bound)
            scale_min = min(scale[0], bound)
 
            target_area = src_h * src_w * np.random.uniform(scale_min,
                                                            scale_max)
            target_size = math.sqrt(target_area)
            w = int(target_size * w)
            h = int(target_size * h)
 
            i = np.random.randint(0, src_w - w + 1)
            j = np.random.randint(0, src_h - h + 1)
 
            img = img[j:j + h, i:i + w]
            img = cv2.resize(img, (resize_w, resize_h))
            return img
