
from PIL import Image
import sys

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import  torch.nn.functional as F
import numpy as np


# 将图片填充为正方形
def fill_image(image):
    width, height = image.size
    # 选取长和宽中较大值作为新图片的
    new_image_length = width if width > height else height
    # 生成新图片[白底]
    new_image = Image.new(image.mode, (new_image_length, new_image_length), color='white')
    # 将之前的图粘贴在新图上，居中
    if width > height:  # 原图宽大于高，则填充图片的竖直维度
        new_image.paste(image, (0, int((new_image_length - height) / 2)))  # (x,y)二元组表示粘贴上图相对下图的起始位置
    else:
        new_image.paste(image, (int((new_image_length - width) / 2), 0))
    return new_image

# 切图
def cut_image(image):
    width, height = image.size
    item_width = int(width / 4)   #(/4)
    #item_width_3 = int(width / 3)   #(/4)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, 4):  # 两重循环，生成9张图片基于原图的位置
        for j in range(0, 4):
            # print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            box_list.append(box)



    image_list = [image.crop(box) for box in box_list]

    return image_list



class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
    #def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        #self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_ = nn.Linear(dim, dim , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        #trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,x_1):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x = x.permute(0,2,1)
        B_, N, C = x.shape
        qkv = self.qkv_(x).reshape(B_, self.num_heads, N,(C // self.num_heads))
        #q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q= qkv
        x_1 = x_1.permute(0,2,1)
        B_, N, C = x_1.shape
        qkv_1 = self.qkv(x_1).reshape(B_, N, 3, self.num_heads, (C // self.num_heads)).permute(2, 0, 3, 1, 4)
        q_1, k_1, v_1 = qkv_1[0], qkv_1[1], qkv_1[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k_1.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v_1).transpose(1, 2).reshape(B_, N, C)  #把维度拉回至和输入样本相同
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




def att_score(image):
    to_tensor = transforms.ToTensor()
    toimg = transforms.ToPILImage()
    image = F.avg_pool2d(image,9)
    # 4 196 9 36
    image = image[0]
    image = toimg(image)
    image_list = cut_image(image)
    att_img = []
    for i in (5,0,3,12,15):
        att_img.append(image_list[i])
    img   = att_img[0]
    img_1 = att_img[1]
    img_2 = att_img[2]
    img_3 = att_img[3]
    img_4 = att_img[4]
    patch_list = [img,img_1,img_2,img_3,img_4]
    img = to_tensor(img)
    img = img.unsqueeze(dim=0)
    img = img.view(img.shape[0], -1, img.shape[1])  # 把（b,c,h,w）沾化辰给（b,h*w,c）
    img_1 = to_tensor(img_1)
    img_1 = img_1.unsqueeze(dim=0)
    img_1 = img_1.view(img_1.shape[0], -1, img_1.shape[1])  # 把（b,c,h,w）沾化辰给（b,
    
    img_2 = to_tensor(img_2)
    img_2 = img_2.unsqueeze(dim=0)
    img_2 = img_2.view(img_2.shape[0], -1, img_2.shape[1])  # 把（b,c,h,w）沾化辰给（b,

    img_3 = to_tensor(img_3)
    img_3 = img_3.unsqueeze(dim=0)
    img_3 = img_3.view(img_3.shape[0], -1, img_3.shape[1])  # 把（b,c,h,w）沾化辰给（b,

    img_4 = to_tensor(img_4)
    img_4 = img_4.unsqueeze(dim=0)
    img_4 = img_4.view(img_4.shape[0], -1, img_4.shape[1])  # 把（b,c,h,w）沾化辰给（b,




    att=WindowAttention(dim=36)  #分割成3*3为1369，4*4的为784

    att_0_1 = att.forward(img, img_1)
    att_0_2 = att.forward(img, img_2)
    att_0_3 = att.forward(img, img_3)
    att_0_4 = att.forward(img, img_4)




    batch = att_0_1.shape[0]

    score = [0,0,0,0]
    score_1 = 0
    score_2 = 0
    score_3 = 0
    score_4 = 0
    

        
    for i in range(batch):
        score[0] += sum(att_0_1[i][0][:])
        score[1] += sum(att_0_2[i][0][:])
        score[2] += sum(att_0_3[i][0][:])
        score[3] += sum(att_0_4[i][0][:])
        
    su = abs(score[0])+abs(score[1])+abs(score[2])+abs(score[3])
    for i in range(4):
        score[i]=abs(score[i]//su)
    return score









