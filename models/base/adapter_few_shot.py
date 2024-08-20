import functools

import torch
from torch.functional import norm
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from collections import OrderedDict
import math
from itertools import combinations
from torch.nn.init import xavier_normal_
import numpy as np
# from torch.nn.modules.activation import MultiheadAttention

from torch.autograd import Variable
import torchvision.models as models
from ipdb import set_trace
from einops import rearrange
import os
from torch.autograd import Variable

from utils.registry import Registry
from models.base.backbone import BACKBONE_REGISTRY
from models.base.base_blocks import HEAD_REGISTRY
# from collections import OrderedDict
from typing import Tuple, Union
import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re

import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

# from .model import build_model
# from .simple_tokenizer import SimpleTokenizer as _Tokenizer
DWCONV3D_DISABLE_CUDNN = True
SUB_FRAME = False

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC



def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out



def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, spatial=False):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.spatial = spatial

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        # if self.spatial:
        #     x_copy = x[1:]
        # x, _ = F.multi_head_attention_forward(
        #     query=x[:1], key=x, value=x,
        #     embed_dim_to_check=x.shape[-1],
        #     num_heads=self.num_heads,
        #     q_proj_weight=self.q_proj.weight,
        #     k_proj_weight=self.k_proj.weight,
        #     v_proj_weight=self.v_proj.weight,
        #     in_proj_weight=None,
        #     in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
        #     bias_k=None,
        #     bias_v=None,
        #     add_zero_attn=False,
        #     dropout_p=0,
        #     out_proj_weight=self.c_proj.weight,
        #     out_proj_bias=self.c_proj.bias,
        #     use_separate_proj_weight=True,
        #     training=self.training,
        #     need_weights=False
        # )
        if self.spatial:
            if self.spatial == "v2":
                cls_token, _ = F.multi_head_attention_forward(
                    query=x[:1], key=x, value=x,
                    embed_dim_to_check=x.shape[-1],
                    num_heads=self.num_heads,
                    q_proj_weight=self.q_proj.weight,
                    k_proj_weight=self.k_proj.weight,
                    v_proj_weight=self.v_proj.weight,
                    in_proj_weight=None,
                    in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                    bias_k=None,
                    bias_v=None,
                    add_zero_attn=False,
                    dropout_p=0,
                    out_proj_weight=self.c_proj.weight,
                    out_proj_bias=self.c_proj.bias,
                    use_separate_proj_weight=True,
                    training=self.training,
                    need_weights=False
                )
                x_mid = self.v_proj(x[1:])
                x_mid = self.c_proj(x_mid)
                x = torch.cat([cls_token, x_mid], dim=0)
                return x.squeeze(0)
            else:

                x, _ = F.multi_head_attention_forward(
                    query=x, key=x, value=x,
                    embed_dim_to_check=x.shape[-1],
                    num_heads=self.num_heads,
                    q_proj_weight=self.q_proj.weight,
                    k_proj_weight=self.k_proj.weight,
                    v_proj_weight=self.v_proj.weight,
                    in_proj_weight=None,
                    in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                    bias_k=None,
                    bias_v=None,
                    add_zero_attn=False,
                    dropout_p=0,
                    out_proj_weight=self.c_proj.weight,
                    out_proj_bias=self.c_proj.bias,
                    use_separate_proj_weight=True,
                    training=self.training,
                    need_weights=False
                )
                # return torch.cat([x, self.c_proj(x_copy)], dim=0)
                return x.squeeze(0)
        else:
            x, _ = F.multi_head_attention_forward(
                query=x[:1], key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.num_heads,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.c_proj.weight,
                out_proj_bias=self.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.training,
                need_weights=False
            )
            return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.用三个stem卷积而不是一个卷积，用最大池化而不是平均池化
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1 执行抗锯齿跨步卷积，其中的平均池化就是给跨步大于1准备的
    - The final pooling layer is a QKV attention instead of an average pool 最后的池化层是qkv注意力，而不是平均池化
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, spatial=False):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim, spatial)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 adapter_width: int,
                 adapter_layers: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=width,
                n_head=heads,
                adapter_width=adapter_width,
                adapter_kernel_size=adapter_kernel_size,
                adapter_pre_attn=adapter_pre_attn and i >= layers - adapter_layers,
                adapter_pre_mlp=adapter_pre_mlp and i >= layers - adapter_layers,
                layer_index=i,
            )
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x, num_frames)
            if num_frames == 7:
                return x
        return x



class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 # dim :768,
                 d_model: int,
                 n_head: int,
                 adapter_width: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 layer_index: int,
                 ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.layer_index = layer_index

        adapter_class = functools.partial(
            Adapter,
            in_channels=d_model,
            adapter_channels=adapter_width,
            kernel_size=adapter_kernel_size,
        )
        # if self.layer_index == 7:
        #     self.conv_1 = nn.Conv3d(
        #         32,  32,
        #         kernel_size=(3,1,1),
        #         stride=(1, 1, 1),
        #         padding=tuple(x // 2 for x in (3,1,1)),
        #         groups=32,
        #     )
        #     self.conv_2 = nn.Conv3d(
        #         32, 8,
        #         kernel_size=(3, 1, 1),
        #         stride=(1, 1, 1),
        #         padding=tuple(x // 2 for x in (3, 1, 1)),
        #         groups=8,
        #     )

            # nn.init.constant_(self.conv_1.weight, 0.)
            # nn.init.constant_(self.conv_1.bias, 0.)
            # nn.init.constant_(self.conv_2.weight, 0.)
            # nn.init.constant_(self.conv_2.bias, 0.)
        self.adapter_pre_attn = \
            adapter_class() if adapter_pre_attn else None
        self.adapter_pre_mlp = \
            adapter_class() if adapter_pre_mlp else None

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.size()
        H = self.attn.num_heads

        qkv = F.linear(x, weight=self.attn.in_proj_weight, bias=self.attn.in_proj_bias)
        qkv = qkv.view(B, L, H * 3, -1).permute(0, 2, 1, 3)
        q, k, v = qkv.split([H, H, H], dim=1)  # 从第二个维度拆分
        # out = F.scaled_dot_product_attention(q, k, v)
        attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1)
        out = torch.matmul(attn_weight , v)#attn_weight @ v
        out = out.permute(0, 2, 1, 3).flatten(-2)
        out = self.attn.out_proj(out)

        return out

    def forward(self,
                x: torch.Tensor,
                num_frames: int
                ) -> torch.Tensor:  # torch.Tensor类型提示
        if self.adapter_pre_attn is not None:
            x = self.adapter_pre_attn(x, num_frames)
        x = x + self.attention(self.ln_1(x))
        if self.layer_index == 7 and SUB_FRAME:
            BT,L,C = x.size()
            B = BT // num_frames
            H = W = round(math.sqrt(L-1))
            assert L - 1 == H * W
            x_id = x
            x = x[:,1:,:]
            x = x.view(B,num_frames,H,W,C).contiguous()
            x1 = x.clone()
            first_image = x[:, 0, :, :, :]
            x1 = torch.cat((first_image.unsqueeze(1), x1), dim=1)
            x1 = torch.cat((first_image.unsqueeze(1), x1), dim=1)
            last_image = x1[:, -1, :, :, :] # 提取最后帧
            x1 = torch.cat((x1, last_image.unsqueeze(1)), dim=1)
            x1 = torch.cat((x1, last_image.unsqueeze(1)), dim=1)
            x_feature = []
            for t in range(2, 10):
                tensor_z1 = x1[:, t, :, :, :] - x1[:, t - 2, :, :, :] # [5,8,14,14,768]
                tensor_z2 = x1[:, t, :, :, :] - x1[:, t - 1, :, :, :]
                tensor_z3 = x1[:, t, :, :, :] - x1[:, t + 1, :, :, :]
                tensor_z4 = x1[:, t, :, :, :] - x1[:, t + 2, :, :, :]
                z_feature = torch.cat((tensor_z1.unsqueeze(1),tensor_z2.unsqueeze(1),tensor_z3.unsqueeze(1),tensor_z4.unsqueeze(1)),dim = 1)
                x_feature.append(z_feature)
            x_feature = torch.stack(x_feature, dim=1)
            x_feature = x_feature.view(B,-1,H,W,C).permute(0,1,4,2,3).contiguous()
            # x_feature = self.conv_1(x_feature)
            x_feature = self.conv_2(x_feature)
            x_feature = x_feature.permute(0, 1,3,4,2).contiguous()
            x = x_feature.view(BT,L-1,C).contiguous()
            x_id[:,1:,:] += x
            x = x_id
            # print("here",self.layer_index)
        if self.adapter_pre_mlp is not None:
            x = self.adapter_pre_mlp(x, num_frames)
        x = x + self.mlp(self.ln_2(x))
        return x

# class Attention(nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_heads=8,
#         qkv_bias=False,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         insert_control_point=False,
#     ):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#
#         self.insert_control_point = insert_control_point
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.n_segment = 8
#         if insert_control_point:
#             self.control_point = AfterReconstruction(dim)
#             self.control_point_query = TemporalShift(dim)
#             self.control_point_value = TemporalShift(dim)
#
#     def forward(self, x):
#         if self.insert_control_point:
#             x = self.control_point(x)
#         B, N, C = x.shape
#
#         qkv = (
#             self.qkv(x)
#             .reshape(B, N, 3, self.num_heads, C // self.num_heads)
#             .permute(2, 0, 3, 1, 4)
#         )
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         if self.insert_control_point:
#             k = self.control_point_query(k)
#             v = self.control_point_value(v)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# class AfterReconstruction(nn.Identity):
#     def __init__(self, inplanes):
#         super().__init__()
#         self.inplanes = inplanes

# class TemporalShift(nn.Module):
#     def __init__(self, dim,n_segment=8, n_div=4):
#         super(TemporalShift, self).__init__()
#
#         self.n_segment = n_segment
#         self.fold_div = n_div
#
#     def forward(self, x):
#         nt, num_heads, d, c = x.size()
#         n_batch = nt // self.n_segment
#         x = x.view(n_batch, self.n_segment, num_heads, d, c)
#         fold = c // self.fold_div
#         out = x.clone()
#         out[:, 1: ,:,  :, :fold] = x[:, :-1, :,  :, :fold]  # shift left
#         out[:, :-1 ,:,  :, fold:2*fold] = x[:, 1:, :,  :, fold:2*fold]  # shift right
#
#         # x = (
#         #     x.permute(0, 1, 2, 4, 3)
#         #     .contiguous()
#         #     .view(n_batch, self.n_segment, num_heads * c, d)
#         # )
#         #
#         # out = torch.zeros_like(x)
#         # out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
#         # out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]  # shift right
#         # out[:, :, 2 * fold :] = x[:, :, 2 * fold :]  # not shift
#
#         # out = (
#         #     out.view(n_batch, self.n_segment, num_heads, c, d)
#         #     .permute(0, 1, 2, 4, 3)
#         #     .contiguous()
#         # )
#
#         return out.view(nt, num_heads, d, c)


class Adapter(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, T):
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels  # 3dcnn
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        x_id = x
        x = x[:, 1:, :]  # [40,196,768]
        x = self.fc1(x)  # 768变384[40,196,384]
        x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2,
                                           3).contiguous()  # [5,8,14,14,384]->[5,384,8,14,14] # 确保张量在内存中是连续存储的

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)  # [384,384,group=384]
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)  # [5,8,14,14,384] ->[40,196,384]
        x = self.fc2(x)  # [384变768]
        x_id[:, 1:, :] += x  # 残差相加除了dim=1 的第一个没加，其他都相加
        return x_id


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int,
                 patch_size: int, #16
                 width: int, #768
                 layers: int,
                 heads: int,
                 output_dim: int,
                 adapter_width: int,
                 adapter_layers: int,  # 12
                 adapter_kernel_size: Tuple[int, int, int],  # 3, 1, 1
                 adapter_pre_attn: bool,  # t
                 adapter_pre_mlp: bool
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5  # 计算了用于初始化嵌入矩阵的缩放因子
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads,
                                         adapter_width, adapter_layers, adapter_kernel_size,
                                         adapter_pre_attn, adapter_pre_mlp)

        self.ln_post = LayerNorm(width)

        for n, p in self.named_parameters():
            if 'adapter' not in n and 'conv_1' not in n and 'conv_2' not in n:  # 将不包含字符串'adapter'的参数设置为不需要梯度计算，并将其数据类型转换为半精度（half）浮点型
                p.requires_grad_(False)
                # if "conv_1"in n or "conv_2"in n:
                #     p.requires_grad_(True)
                p.data = p.data.half()  # 减少模型计算

        self.dropout = nn.Dropout(0.5)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        # nn.init.normal_(self.fc.weight, std=0.02)
        # nn.init.constant_(self.fc.bias, 0.)

    def forward(self, x: torch.Tensor):

        B, T = x.size(0), x.size(2)
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        x = self.conv1(x)  # shape = [*, width, grid, grid][40,768,14,14]
        spatial_size = tuple(x.size()[2:])
        x = x.flatten(-2).permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1), x
        ], dim=1)  # [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.view(B, T, x.size(1), x.size(2)).flatten(0, 1)  # BT, L, D

        x = self.transformer(x, T)

        x = x.contiguous().view(B, T, spatial_size[0] * spatial_size[1] + 1, x.size(-1))  # [5,8,197,768]
       # x =   # [16,8,768]->[16,768]

        x = self.ln_post(x[:, :, 0, :]) #5,8,768
        x = self.dropout(x)

        if self.proj is not None:
            x = x @ self.proj

        return x


