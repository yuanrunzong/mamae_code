import torch
import numpy as np

def prduct(p,h,w,x):
    mask_p = np.random.choice([1, 0], size=[h//2 , w //2], p=[abs(p), 1 - abs(p)])
    mask_p = torch.tensor(mask_p,device = x.device)

    return mask_p
