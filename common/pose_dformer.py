import math
import random
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
from torch.nn.init import constant_
from diffusers.models.modeling_utils import ModelMixin


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DeformableBlock(nn.Module):
    def __init__(self, dim, num_heads, num_samples, qkv_bias=False, drop_path=0., mlp_ratio=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.num_samples = num_samples
        head_dim = dim // num_heads
        self.norm1 = norm_layer(dim)
        self.attention_weights = nn.Linear(dim, num_heads * num_samples)
        self.sampling_offsets = nn.Linear(dim, 2 * num_heads * num_samples)
        self.embed_proj = nn.ModuleList([
            nn.Linear(32, head_dim),
            nn.Linear(64, head_dim),
            nn.Linear(128, head_dim),
            nn.Linear(256, head_dim),
        ])

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

        self._reset_parameters()

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = 0.01 * (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 2).repeat(1, self.num_samples, 1)
        for i in range(self.num_samples):
            grid_init[:, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

    def forward(self, x, ref, features_list):
        x_0, x = x[:, :1], x[:, 1:]
        b, l, p, c = x.shape
        residual = x
        x = self.norm1(x + x_0)

        weights = self.attention_weights(x).view(b, l, p, self.num_heads, self.num_samples)
        weights = F.softmax(weights, dim=-1).unsqueeze(-1) # b, l, p, num_heads, num_samples, 1
        offsets = self.sampling_offsets(x).reshape(b, l, p, self.num_heads*self.num_samples, 2).tanh()
        pos = offsets + ref.view(b, 1, p, 1, -1)

        features_sampled = [
            F.grid_sample(features, pos[:, idx], padding_mode='border', align_corners=True).permute(0, 2, 3, 1).contiguous() \
            for idx, features in enumerate(features_list)]

        # b, p, num_heads*num_samples, c
        features_sampled = [embed(features_sampled[idx]) for idx, embed in enumerate(self.embed_proj)]
        features_sampled = torch.stack(features_sampled, dim=1) # b, l, p, num_heads*num_samples, c // num_heads

        features_sampled = (weights * features_sampled.view(b, l, p, self.num_heads, self.num_samples, -1)).sum(dim=-2).view(b, l, p, -1)
        
        x = residual + self.drop_path(features_sampled)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = torch.cat([x_0,x], dim=1)
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PoseTransformer(ModelMixin, nn.Module):
    def __init__(self, config=None, num_frame=1, num_joints=17, in_chans=2, 
                 mask_pose=False, mask_context=False, shuffle_context=False,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        self.mask_pose = mask_pose
        self.mask_context = mask_context
        self.shuffle_context = shuffle_context

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim_ratio = 128
        depth = 4
        out_dim = 3
        self.levels = 4
        embed_dim = embed_dim_ratio * (self.levels + 2)

        # spatial patch embedding
        self.coord_3d_embed = nn.Linear(3, embed_dim_ratio)
        self.coord_embed = nn.Linear(in_chans, embed_dim_ratio)
        self.feat_embed = nn.ModuleList([
            nn.Linear(32, embed_dim_ratio),
            nn.Linear(64, embed_dim_ratio),
            nn.Linear(128, embed_dim_ratio),
            nn.Linear(256, embed_dim_ratio),
        ])

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, 1+self.levels, num_joints, embed_dim_ratio))

        # timestep embedding for diffusion model
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, embed_dim_ratio*2),
            nn.GELU(),
            nn.Linear(embed_dim_ratio*2, embed_dim_ratio),
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.joint_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.res_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.context_blocks = nn.ModuleList([
            DeformableBlock(dim=embed_dim_ratio, num_heads=4, num_samples=4, qkv_bias=qkv_bias, drop_path=dpr[i])
            for i in range(depth)])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def forward(self, keypoints_2d, ref, noisy_keypoints_3d, features_list, t):
        # concatenate 2D keypoints and 3D noisy keypoints and embed them
        if len(noisy_keypoints_3d.shape) > 4:
            keypoints_2d = keypoints_2d[:, None].repeat(1, noisy_keypoints_3d.shape[1], 1, 1, 1)
            b, h, f, p, c = keypoints_2d.shape
            levels = self.levels+2
            x = rearrange(keypoints_2d, 'b h f p c -> (b h f) p c')
            x = self.coord_embed(x)
            x_3d_noisy = rearrange(noisy_keypoints_3d, 'b h f p c -> (b h f) p c')
            x_3d_noisy = self.coord_3d_embed(x_3d_noisy)
            x_3d_noisy = rearrange(x_3d_noisy, '(b h f) p c -> (b h) f p c', b=b, h=h)
            time_embed = self.time_mlp(t)[:, None, None, None, :].repeat(1, h, levels, p, 1)
            time_embed = rearrange(time_embed, 'b h f p c -> (b h) f p c')
        else:
            b, f, p, c = keypoints_2d.shape
            levels = self.levels+2
            x = rearrange(keypoints_2d, 'b f p c -> (b f) p c')
            x = self.coord_embed(x)
            x_3d_noisy = rearrange(noisy_keypoints_3d, 'b f p c -> (b f) p c')
            x_3d_noisy = self.coord_3d_embed(x_3d_noisy)[:, None, :, :]
            time_embed = self.time_mlp(t)[:, None, None, :].repeat(1, levels, p, 1)

        # embed context features sampled from reference points
        features_ref_list = [
            F.grid_sample(features, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous() \
            for features in features_list
        ]
        features_ref_list = [embed(features_ref_list[idx]) for idx, embed in enumerate(self.feat_embed)]
        if len(noisy_keypoints_3d.shape) > 4:
            features_ref_list = [features_ref_list[idx][:, None, None, :, :] for idx in range(len(features_ref_list))]
            features_ref_list = [features_ref_list[idx].repeat(1, h, f, 1, 1) for idx in range(len(features_ref_list))]
            features_ref_list = [rearrange(features_ref_list[idx], 'b h f p c -> (b h f) p c') for idx in range(len(features_ref_list))]

        # concatenate context features with 2D poses and 3D noisy poses
        x = torch.stack([x, *features_ref_list], dim=1)

        # add spatial position embedding
        x += self.Spatial_pos_embed

        # add dropout
        x = self.pos_drop(x)

        if len(noisy_keypoints_3d.shape) > 4:
            ref = ref[:, None].repeat(1, noisy_keypoints_3d.shape[1], 1, 1)
            ref = rearrange(ref, 'b h p c -> (b h) p c')
            ref = ref[:, None, :, None, :]
            features_list = [
                rearrange(features_list[idx][:, None, :, :, :].repeat(1, h, 1, 1, 1), 'b h f p c -> (b h) f p c') 
                for idx in range(len(features_list))
            ]
        for blk in self.context_blocks:
            x = blk(x, ref, features_list)

        if self.mask_pose:
            x = torch.cat((torch.zeros_like(x[:, 0:1]).to(x.device), x[:, 1:]), dim=1)
        if self.mask_context:
            x = torch.cat((x[:, 0:1], torch.zeros_like(x[:, 1:]).to(x.device)), dim=1)
        if self.shuffle_context:
            x = torch.cat((x[:, 0:1], x[:, 1:][:, torch.randperm(x[:, 1:].size(1))]), dim=1)

        # concatenate 3D noisy keypoints embedding with context-aware embedding
        x = torch.cat((x_3d_noisy, x), dim=1)
        # add timestep embedding
        x += time_embed
        x = rearrange(x, 'b l p c -> (b p) l c')

        #attention between context and pose
        for blk in self.res_blocks:
            x = blk(x)
        if len(noisy_keypoints_3d.shape) > 4:
            x = rearrange(x, '(b p) l c -> b p (l c)', b=b*h)
        else:
            x = rearrange(x, '(b p) l c -> b p (l c)', b=b)

        #attention between joints
        for blk in self.joint_blocks:
            x = blk(x)

        #noise prediction
        x = self.head(x)

        if len(noisy_keypoints_3d.shape) > 4:
            x = rearrange(x, '(b h) p c -> b h p c', b=b).unsqueeze(2)
        else:
            x = rearrange(x, '(b f) p c -> b f p c', b=b)
        
        return x
