from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed
from util.img_9_you import att_score
from torchvision import transforms
from util.new_view import RGB2Lab, RGB2YCbCr, RGB2HED, RGB2LUV, img_to_pil
import matplotlib.pyplot as plt
import numpy as np


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    ///
    """
                
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 
                 
                ):
        super().__init__()
        decoder_embed_dim=768
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        '''
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        '''

        self.encoder1 = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
       
        self.encoder2 = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        # MAE decoder specifics
        #self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)#_————————————————
        
        
        self.decoder_embed = nn.Linear(embed_dim, 768, bias=True)
        self.ids_restore2 = nn.Linear(768 , 512,bias = False)
    
        #self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))  # parameter的作用将tonser转化为
        # 可训练的参数加入到moudle中
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 768)) 
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
      

        self.norm_pix_loss = norm_pix_loss  # 是否对像素进行归一化后再计算Loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _momentum_update_encoder2_blocks(self):
        """
        Momentum update of the key encoder
        """
        for param_1, param_2 in zip(self.encoder1.parameters(), self.encoder2.parameters()):
            param_2.data = param_2.data * 0.1 + param_1.data * (1. - 0.1)
            #param_1.data = param_1.data * 0.9 + param_2.data * (1. - 0.1)

    # ————————————————————————————————————————————————

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3) l是划分的ptach数量
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        #         print("查看Shape形状")
        #         print(p)
        #         print(imgs.shape)
        #         print(x.shape)
        return x
    
    def encoder1_patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        toimg = transforms.ToPILImage()
        img = imgs[0]
        img = toimg(img)
        img = RGB2Lab(img)
        img = img_to_pil(img)
        imgs[0][:] = img

        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x_1 = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x_1 = torch.einsum('nchpwq->nhwpqc', x_1)
        x_1 = x_1.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x_1

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def get_att_score(self, imgs):
        batch, _, h, w = imgs.shape
 
        att_mask_list = np.zeros((batch, 4))
        for i in range(batch):
            x_img = imgs[i]
            att_mask_list[i][0], att_mask_list[i][1], att_mask_list[i][2], att_mask_list[i][3] = att_score(x_img)
        return att_mask_list, h, w
    

    def attention_masking(self, x, imgs, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim



        att_mask_list, h, w = self.get_att_score(imgs)  # att_mask_list :(batch,4)
        
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, D,device=x.device)
        all_mask = np.zeros((h, w))
        all_mask = all_mask[:, :, np.newaxis]
        for i in range(N):
            p_0, p_1, p_2, p_3 = att_mask_list[i][0], att_mask_list[i][1], att_mask_list[i][2], att_mask_list[i][3]

            mask_p0 = np.random.choice([1, 0], size=[h // 2, w // 2], p=[abs(p_0), 1 - abs(p_0)])
            mask_p1 = np.random.choice([1, 0], size=[h // 2, w // 2], p=[abs(p_1), 1 - abs(p_1)])
            mask_p2 = np.random.choice([1, 0], size=[h // 2, w // 2], p=[abs(p_2), 1 - abs(p_2)])
            mask_p3 = np.random.choice([1, 0], size=[h // 2, w // 2], p=[abs(p_3), 1 - abs(p_3)])

            mask_p0 = np.concatenate((mask_p0, mask_p1), axis=1)  # axis=1表示横向拼接
            mask_p2 = np.concatenate((mask_p2, mask_p3), axis=1)  # axis=1表示横向拼接
            mask_p = np.concatenate((mask_p0, mask_p2), axis=0)  # axis=0表示纵向拼接


            mask_p = mask_p[:, :, np.newaxis] 

            all_mask = np.concatenate((all_mask, mask_p), axis=2)
        
        all_mask = np.delete(all_mask, [0], 2)

        all_mask = all_mask.transpose(2, 0, 1).reshape(N, L, -1)
        
        _, _, leng = all_mask.shape
        for i in range(leng - 1):
            all_mask = np.delete(all_mask, [0], 2)

        all_mask = torch.from_numpy(all_mask)  
        all_mask = all_mask.cuda()
        noise = torch.mul(noise, all_mask)

        '''
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        '''
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        #x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) 
        x_masked = torch.gather(x, dim=1, index=ids_keep)  #
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L,D], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore  # x_mask (N,len_leep,D) mask (N,L)  ids_restore(N,L)




    def forward_encoder(self, x, x_1, mask_ratio, imgs):
        # embed patches
        x = self.patch_embed(x)
        x_1 = self.patch_embed(x_1)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        x_1 = x_1 + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x_1, mask, ids_restore = self.attention_masking(x_1, imgs, mask_ratio)
        x, mask, ids_restore = self.attention_masking(x, imgs, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        cls_tokens = cls_token.expand(x_1.shape[0], -1, -1)
        x_1 = torch.cat((cls_tokens, x_1), dim=1)

        # apply Transformer blocks
        for blk in self.encoder1:
            x = blk(x)

        
        x = self.norm(x)

        with torch.no_grad():  # no gradient to encoder2
            self._momentum_update_encoder2_blocks()
        for blk in self.encoder2:
            x_1 = blk(x_1)
        x_1 = self.norm(x_1)

        return x, x_1, mask, ids_restore



    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x) 

        # append mask tokens to sequence
        #self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        #ids_restore(N,L)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        #将得到的（1，1，decoder_dim）全0的数据改成与x维度相同,即mask_tokens
        
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        #将mask_tokens与编码器的输出相拼接
        
        ids_restore.type(torch.int64)
        x_ = torch.gather(x_, dim=1, index=ids_restore)
        #按照ids_retore将掩码部分与编码器输出顺序用gather恢复
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, pred_1, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        '''
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
        '''
        target = self.patchify(imgs)
        target_1 = self.encoder1_patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.mean(dim=-1) 
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss_1 = (pred_1 - target_1) ** 2
        loss_1 = loss_1.mean(dim=-1)  # [N, L], mean loss per patch
        loss_1 = (loss_1 * mask).sum() / mask.sum()  # mean loss on removed patches
        loss = 0.90 * loss + 0.1 * loss_1
        return loss

    


    def forward(self, imgs, mask_ratio=0.75, clip_teacher=None, cidx=None, cluster_result=None):
        latent, latent_1, mask, ids_restore = self.forward_encoder(imgs, imgs, mask_ratio, imgs)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        pred_1 = self.forward_decoder(latent_1, ids_restore)
        loss = self.forward_loss(imgs, pred, pred_1, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
