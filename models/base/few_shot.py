import functools

import torch
from torch.functional import norm
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
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
from models.base.pytorch_i3d import InceptionI3d
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
from pkg_resources import packaging

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from models.base.adapter_few_shot import VisionTransformer
# from .model import build_model
# from .simple_tokenizer import SimpleTokenizer as _Tokenizer
DWCONV3D_DISABLE_CUDNN = True
AddPrompt = False
OPTICAL = True
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


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


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
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

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = SimpleTokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


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


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

def InceptionI3d_load(device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
    # model_path = _download("https://openaipublic.azureedge.net/clip/models/b8cca3/flow_charades.pt",os.path.expanduser("~/.cache/clip/adpter"))
    model = torch.load("/home/imi1214/.cache/clip/adpter/flow_charades.pt")
    i3d = InceptionI3d(400, in_channels=2)
    i3d.replace_logits(157)
    i3d.load_state_dict(model)
    i3d.cuda()
    return i3d

def load(name: str, cfg, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None, spatial=False):
    """Load a CLIP model
    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).
    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive  pytorch解释器版本要高一点
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens) # 将其赋值给 result 的第 i 行的前 len(tokens) 列。

    return result





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
            if self.spatial=="v2":
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



class TransformerAD(nn.Module):
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
            ResidualAttentionBlockAD(
                d_model=width,
                n_head=heads,
                adapter_width=adapter_width,
                adapter_kernel_size=adapter_kernel_size,
                adapter_pre_attn=adapter_pre_attn and i >= layers - adapter_layers,
                adapter_pre_mlp=adapter_pre_mlp and i >= layers - adapter_layers,
            )
            for i in range(layers)
        ])
    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x, num_frames)
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ResidualAttentionBlockAD(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 adapter_width: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
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

        adapter_class = functools.partial(
            Adapter,
            in_channels=d_model,
            adapter_channels=adapter_width,
            kernel_size=adapter_kernel_size,
        )
        self.adapter_pre_attn = \
            adapter_class() if adapter_pre_attn else None
        self.adapter_pre_mlp = \
            adapter_class() if adapter_pre_mlp else None

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.size()
        H = self.attn.num_heads

        qkv = F.linear(x, weight=self.attn.in_proj_weight, bias=self.attn.in_proj_bias)
        qkv = qkv.view(B, L, H * 3, -1).permute(0, 2, 1, 3)
        q, k, v = qkv.split([H, H, H], dim=1)# 从第二个维度拆分
        #out = F.scaled_dot_product_attention(q, k, v)
        attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1)
        out = attn_weight @ v
        out = out.permute(0, 2, 1, 3).flatten(-2)
        out = self.attn.out_proj(out)

        return out

    def forward(self,
                x: torch.Tensor,
                num_frames: int
                ) -> torch.Tensor: # torch.Tensor类型提示
        if self.adapter_pre_attn is not None:
            x = self.adapter_pre_attn(x, num_frames)
        x = x + self.attention(self.ln_1(x))
        if self.adapter_pre_mlp is not None:
            x = self.adapter_pre_mlp(x, num_frames)
        x = x + self.mlp(self.ln_2(x))
        return x
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
        Ca = self.conv.in_channels # 3dcnn
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        x_id = x
        x = x[:, 1:, :] # [512,196,768]
        x = self.fc1(x) # 768变384[512,196,384]
        x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous() #[64,8,14,14,384]->[64,384,8,14,14] # 确保张量在内存中是连续存储的

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x) # [384,384,group=384]
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca) # [64,8,14,14,384] ->[512,196,384]
        x = self.fc2(x) # [384变768]
        x_id[:, 1:, :] += x # 残差相加除了dim=1 的第一个没加，其他都相加
        return x_id



class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 adapter_width: int,  # 384
                 adapter_layers: int,  # 12
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 spatial=False,

                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,   # (3, 4, 6, 3)
                output_dim=embed_dim,   # 1024
                heads=vision_heads,     # 32
                input_resolution=image_resolution,    # 224
                width=vision_width,      # 64
                spatial=spatial
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution, # 224
                patch_size=vision_patch_size, # 16
                width=vision_width, # 768
                layers=vision_layers, # 12
                heads=vision_heads, # 12
                output_dim=embed_dim, # 512
                adapter_width=adapter_width,# 384
                adapter_layers=adapter_layers, # 12
                adapter_kernel_size=adapter_kernel_size, # 3,1,1
                adapter_pre_attn=adapter_pre_attn,# t
                adapter_pre_mlp=adapter_pre_mlp # T

            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) # 一个标量值为 1 的张量,NumPy 的 log 函数计算了 1 / 0.07 的自然对数。这个计算结果表示一个缩放因子，

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std) # k
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std) # q
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std) # v
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters(): # 将模型中所有以 "bn3.weight" 结尾的参数的值初始化为零
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks: # 函数对神经网络模型中的权重进行正态分布初始化操作
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std) # 用block.attn.in_proj_weight的值正态分布初始化，并指定初始值
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):

        return torch.float32

    def encode_text_light(self, text):
        x_light = self.token_embedding(text).type(self.dtype)   # [batch_size, n_ctx, d_model]
        return x_light

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    # def encode_text(self,xlight, text):
    #    # x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
    #
    #     x = xlight + self.positional_embedding.type(self.dtype)
    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     x = self.transformer(x)
    #     x = x.permute(1, 0, 2)  # LND -> NLD
    #     x = self.ln_final(x).type(self.dtype)
    #
    #     # x.shape = [batch_size, n_ctx, transformer.width]
    #     # take features from the eot embedding (eot_token is the highest number in each sequence)
    #     x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    #
    #     return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, spatial=False):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else: # set 除去重复值，len 取长度
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0] # 64 第一层第一个卷积的权重
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5) # 输出宽度
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0] # 断言上述又没有算错
        image_resolution = output_width * 32 # 图像分辨率

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64 # 8个头
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    adapter_width = 576
    adapter_layers = 12
    adapter_kernel_size = (3, 1, 1)
    adapter_pre_attn = True
    adapter_pre_mlp = True

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        adapter_width, adapter_layers, adapter_kernel_size, adapter_pre_attn, adapter_pre_mlp,
        spatial=spatial,

    )

    for key in ["input_resolution", "context_length", "vocab_size"]: # 删除指定键 为了在加载模型时排除或修改某些特定参数。
        if key in state_dict:
            del state_dict[key]
    # for n, p in model.named_parameters():
    #     if 'adapter' not in n:  # 将不包含字符串'adapter'的参数设置为不需要梯度计算，并将其数据类型转换为半精度（half）浮点型
    #         p.requires_grad_(False)
    #         p.data = p.data.half()
    # convert_weights(model) # 将模型中所有适用的参数转换为半精度浮点数，以便后续在半精度计算环境中进行训练或推理
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


class Up2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=kernel_size, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2(in_channels, out_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=kernel_size, groups=1)
            self.conv = DoubleConv2(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=8),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class PreNormattention_qkv(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, q, k, v, **kwargs):
        return self.fn(self.norm(q), self.norm(k), self.norm(v), **kwargs) + q

class Transformer_v1(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte = 0.05, mlp_dim=2048, dropout_ffn = 0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                PreNormattention_qkv(dim, Attention_qkv(dim, heads = heads, dim_head = dim_head_k, dropout = dropout_atte)),
                FeedForward(dim, mlp_dim, dropout = dropout_ffn),
            ]))
    def forward(self, q, k, v):
        # if self.depth
        for attn, ff in self.layers[:1]:
            # x = q + attn(q, k, v)
            x = attn(q, k, v)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x, x, x)
                x = ff(x) + x
        return x

class Transformer_v2(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte = 0.05, mlp_dim=2048, dropout_ffn = 0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                PreNormattention(dim, Attention(dim, heads = heads, dim_head = dim_head_k, dropout = dropout_atte)),
                FeedForward(dim, mlp_dim, dropout = dropout_ffn),
            ]))
    def forward(self, x):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(x)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x)
                x = ff(x) + x
        return x


class PreNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x




class Attention_qkv(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads # 【5，8，512】
        bk = k.shape[0] # 第一个维度25或者5
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        # q = rearrange(q, '(b n) q (h d) -> b h (n q) d', b=bk, h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', b=bk, h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', b=bk, h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)    # [30, 8, 8, 5]

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # if out.size(2) == 9:
        #     out = rearrange(out, 'b h n d -> b n (h d)')
        # else:
        #     out = rearrange(out, 'b h (n q) d -> (b n) q (h d)', b=bk, h=h, q=8)
        return self.to_out(out)

class Attention_qkv_text(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, h = *q.shape, self.heads
        bk = k.shape[0] # 第一个维度25或者5

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        q = rearrange(q, 'b (h d) -> b h d', h=h)
        k = rearrange(k, 'b (h d) -> b h d', b=bk, h=h)
        v = rearrange(v, 'b (h d) -> b h d', b=bk, h=h)
        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, 'b n d -> b (n d)')

        return self.to_out(out)


class PostNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs) + x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists


def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector



class CNN_FSHead(nn.Module):
    """
    Base class which handles a few-shot method. Contains a resnet backbone which computes features.
    """
    def __init__(self, cfg):
        super(CNN_FSHead, self).__init__()
        args = cfg
        self.train()
        self.args = args

        last_layer_idx = -1


    def get_feats(self, support_images, target_images):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()
        target_features = self.backbone(target_images).squeeze()

        dim = int(support_features.shape[1])

        support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)

        return support_features, target_features

    def forward(self, support_images, support_labels, target_images):
        """
        Should return a dict containing logits which are required for computing accuracy. Dict can also contain
        other info needed to compute the loss. E.g. inter class distances.
        """
        raise NotImplementedError

    def distribute_model(self):
        """
        Use to split the backbone evenly over all GPUs. Modify if you have other components
        """
        if self.args.TRAIN.DDP_GPU > 1:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(0, self.args.TRAIN.DDP_GPU)])
    
    def loss(self, task_dict, model_dict):
        """
        Takes in a the task dict containing labels etc.
        Takes in the model output dict, which contains "logits", as well as any other info needed to compute the loss.
        Default is cross entropy loss.
        """
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
        
        


class PositionalEncoding(nn.Module):
    """
    Positional encoding from the Transformer paper.
    """
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)


def OTAM_cum_dist(dists, lbda=0.1):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len] 
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1,1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:,:,0,m] = dists[:,:,0,m] + cum_dists[:,:,0,m-1] 


    # remaining rows
    for l in range(1,dists.shape[2]):
        #first non-zero column
        cum_dists[:,:,l,1] = dists[:,:,l,1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,0] / lbda) + torch.exp(- cum_dists[:,:,l-1,1] / lbda) + torch.exp(- cum_dists[:,:,l,0] / lbda) )
        
        #middle columns
        for m in range(2,dists.shape[3]-1):
            cum_dists[:,:,l,m] = dists[:,:,l,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,m-1] / lbda) + torch.exp(- cum_dists[:,:,l,m-1] / lbda ) )
            
        #last column
        #cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l-1,-1] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
    
    return cum_dists[:,:,-1,-1]

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class PositionalEncoder(nn.Module):
    def __init__(self, d_model=2048, max_seq_len = 20, dropout = 0.1, A_scale=10., B_scale=1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.A_scale = A_scale
        self.B_scale = B_scale
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        
        x = x * math.sqrt(self.d_model/self.A_scale)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + self.B_scale * pe
        return self.dropout(x)


def OTAM_cum_dist_v2(dists, lbda=0.5):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len] 
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1,1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:,:,0,m] = dists[:,:,0,m] + cum_dists[:,:,0,m-1] 


    # remaining rows
    for l in range(1,dists.shape[2]):
        #first non-zero column
        cum_dists[:,:,l,1] = dists[:,:,l,1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,0] / lbda) + torch.exp(- cum_dists[:,:,l-1,1] / lbda) + torch.exp(- cum_dists[:,:,l,0] / lbda) )
        
        #middle columns
        for m in range(2,dists.shape[3]-1):
            cum_dists[:,:,l,m] = dists[:,:,l,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,m-1] / lbda) + torch.exp(- cum_dists[:,:,l,m-1] / lbda ) )
            
        #last column
        #cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l-1,-1] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
    
    return cum_dists[:,:,-1,-1]




class Transformer_flow(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dropout_atte = 0.05, mlp_dim=2048, dropout_ffn = 0.05):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(
                nn.ModuleList([
                    PreNormattention_qkv(dim, Attention_qkv_text(dim, heads=heads, dim_head=dim_head_k, dropout=dropout_atte)),
                    FeedForward(dim, mlp_dim, dropout=dropout_ffn),
        ]))

    def forward(self, q, k, v):
    # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(q, k, v)
            x = ff(x) + x
        return x
# class Transformer_flow(nn.Module):
#     def __init__(self, width: 512, layers: 1, heads: int, attn_mask: torch.Tensor = None):
#         super().__init__()
#         self.width = width
#         self.layers = layers
#         self.resblocks= ResidualAttentionBlock(width, heads, attn_mask)
#
#
#     def forward(self, x: torch.Tensor):
#         return self.resblocks(x)

# 单行线光流
@HEAD_REGISTRY.register()
class CLIP_ADAPTER_24(CNN_FSHead):
    def __init__(self, cfg):
        super(CLIP_ADAPTER_24, self).__init__(cfg) # 进入到CNN_FSHead的初始化里面
        args = cfg
        self.args = cfg
        self.embedding = torch.nn.Embedding(77, 512)
        nn.init.normal_(self.embedding.weight, std=0.01)
        if cfg.VIDEO.HEAD.BACKBONE_NAME == "ViT-B/16":
            backbone, self.preprocess = load(cfg.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=cfg, jit=False)
            self.text = backbone.encode_text  # ViT-B/16
            self.backbone = backbone.visual  # model.load_state_dict(state_dict)
            self.class_real_train = cfg.TRAIN.CLASS_NAME
            self.class_real_test = cfg.TEST.CLASS_NAME
            self.mid_dim = 512
            self.opticalFlow_branch = InceptionI3d_load(device="cuda")
            self.fc_sub = nn.Linear(1024, self.mid_dim)
            nn.init.constant_(self.fc_sub.bias, 0.)
            # self.opticalFlow_branch.load_state_dict(torch.load(load_model))

        else:
            print("error")
        with torch.no_grad():
            if AddPrompt == True:

                text_templete = [(self.class_real_train[int(ii)]) for ii in range(len(self.class_real_train))]
                self.text_templete_train = tokenize(text_templete).cuda()  # 把64个class_name全部变成tensor[64,77]
                self.text_embedding_train = backbone.encode_text_light(self.text_templete_train)  # [64,1024]
                text_templete = [(self.class_real_test[int(ii)]) for ii in range(len(self.class_real_test))]
                self.text_templete_test = tokenize(text_templete).cuda()
                self.text_embedding_test = backbone.encode_text_light(self.text_templete_test)
                self.actiondict_train = OrderedDict((self.class_real_train[i], self.text_embedding_train[i].cuda().cpu().data.numpy()) for i in range(len(self.class_real_train)))
                self.actiondict_test = OrderedDict(
                    (self.class_real_test[i], self.text_embedding_test[i].cuda().cpu().data.numpy()) for i in
                    range(len(self.class_real_test)))
                self.actiontoken_train = OrderedDict(
                    (self.class_real_train[i], self.text_templete_train[i]) for i in range(len(self.class_real_train)))
                self.actiontoken_test = OrderedDict(
                    (self.class_real_test[i], self.text_templete_test[i]) for i in range(len(self.class_real_test)))
            else:
                text_templete = ["a photo of {}".format(self.class_real_train[int(ii)]) for ii in range(len(self.class_real_train))]
                text_templete = tokenize(text_templete).cuda()  # 把64个句子全部变成token
                self.text_features_train = backbone.encode_text(text_templete)

                text_templete = ["a photo of {}".format(self.class_real_test[int(ii)]) for ii in range(len(self.class_real_test))]
                text_templete = tokenize(text_templete).cuda()

                self.text_features_test = backbone.encode_text(text_templete)

        # self.decoder = GRU()
        self.mid_layer = nn.Sequential()  # 定义中间层
        self.classification_layer = nn.Sequential()  # 定义分类层
        self.scale = nn.Parameter(torch.FloatTensor(1),requires_grad=True)  # 定义一个学习参数learnable parameter，它是一个浮点数的张量（torch.FloatTensor）。
        self.scale.data.fill_(1.0)
        # self.scale_2 = nn.Parameter(torch.FloatTensor(1),requires_grad=True)  # 定义一个学习参数learnable parameter，它是一个浮点数的张量（torch.FloatTensor）。
        # self.scale_2.data.fill_(1.0)
        # nn.Parameter 函数用于将张量转换为可学习的参数。这个参数的初始值被填充为 1.0

        if hasattr(self.args.TRAIN, "TRANSFORMER_DEPTH") and self.args.TRAIN.TRANSFORMER_DEPTH:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2,
                                           depth=int(self.args.TRAIN.TRANSFORMER_DEPTH))
        else:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2)
        # self.flowTrans  = Transformer_flow(width=self.mid_dim, layers=1, heads=8, attn_mask=None)
        self.flowTrans = Transformer_flow(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2)
        # self.mha = MultiheadAttention(embed_dim=self.mid_dim, num_heads=8, dropout=0.2)
    def process_prompt(self, class_name, actiondict, actiontoken, prefix=16, postfix=16):
        self.text_embedding = self.embedding(torch.arange(77).cuda())[None, :].repeat([len(class_name), 1, 1]).to(
            "cuda")  # [64,77,512]
        self.prompt_actiontoken = torch.zeros(len(class_name), 77)

        for i, a in enumerate(class_name):
            emb = torch.from_numpy(actiondict[a]).float().to("cuda")  # [77,512]
            token = torch.from_numpy(np.array(actiontoken[a].cpu().numpy()))  # [77]
            self.text_embedding[i][0] = emb[0]  # [512]
            ind = np.argmax(token, -1)

            self.text_embedding[i][prefix + 1:prefix + ind] = emb[1:ind]  # [17:19]=[1,3]
            self.text_embedding[i][prefix + ind + postfix] = emb[ind]  # [33]=[3]

            self.prompt_actiontoken[i][0] = token[0]  # []
            self.prompt_actiontoken[i][prefix + 1:prefix + ind] = token[1:ind]  # [17:19]=[1:3]
            self.prompt_actiontoken[i][prefix + ind + postfix] = token[ind]

        self.text_embedding.to("cuda")
        self.prompt_actiontoken.to("cuda")

    def get_feats(self, support_images, target_images, support_real_class=False, support_labels=False):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        if self.training:

            support_features = self.backbone(support_images.permute(0,2,1,3,4)).squeeze()
            # os.system("nvidia-smi")
            target_features = self.backbone(target_images.permute(0,2,1,3,4)).squeeze() #[5,8,1024]

            dim = int(support_features.shape[1])  # [2048]

            # Query_num = target_features.shape[0] // self.args.DATA.NUM_INPUT_FRAMES  # [5]
            # support_features = self.relu(self.temporal_atte_before(self.pe(support_features)))  # [35, 5, 8, 2048]  V1
            # target_features = self.relu(self.temporal_atte_before(self.pe(target_features)))
            # class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, 64)

            # os.system("nvidia-smi")

            # dim = int(support_features.shape[1])
            #
            # support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            # target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            support_features_text = None

        else:
            support_features = self.backbone(support_images.permute(0,2,1,3,4)).squeeze()
            target_features = self.backbone(target_images.permute(0,2,1,3,4)).squeeze()
            # dim = int(target_features.shape[1])
            # target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            # support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            # support_real_class = torch.unique(support_real_class)
            support_features_text = self.text_features_test[support_real_class.long()]

        return support_features, target_features, support_features_text

    def forward(self, inputs):
        support_images, support_labels, target_images, support_real_class,batch_class = inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['real_support_labels'],inputs['batch_class_list']  # [200, 3, 224, 224] inputs["real_support_labels"]
        support_sub_frame,target_sub_frame = inputs['support_subframe_set'],inputs['target_subframe_set']
        # set_trace()
        if AddPrompt == True:

            self.process_prompt(self.class_real_train, self.actiondict_train, self.actiontoken_train)
            self.text_features_train = self.text(self.text_embedding, self.prompt_actiontoken)
            self.process_prompt(self.class_real_test, self.actiondict_test, self.actiontoken_test)
            self.text_features_test = self.text(self.text_embedding, self.prompt_actiontoken)
        else:
            pass
        if self.training:
            support_features, target_features, _ = self.get_feats(support_images, target_images,support_labels)
            if OPTICAL == True:# [5,8,1024]
                with torch.no_grad():
                    support_sub_feature = self.opticalFlow_branch(support_sub_frame.permute(0,2,1,3,4))
                    target_sub_feature = self.opticalFlow_branch(target_sub_frame.permute(0,2,1,3,4))
                support_sub_feature = self.fc_sub(support_sub_feature)
                support_sub_feature= self.flowTrans(support_sub_feature,support_sub_feature, support_sub_feature)
                target_sub_feature = self.fc_sub(target_sub_feature)
                target_sub_feature=  self.flowTrans(target_sub_feature,target_sub_feature,target_sub_feature)

            if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
                feature_classification_in = torch.cat([support_features, target_features],dim=0)  # 在0维度上进行cat【30，8，512】
                # feature_classification_flow = torch.cat([support_sub_feature, target_sub_feature],dim=0) # [30]
                feature_classification = self.classification_layer(feature_classification_in).mean(1)

                class_text_logits = cos_sim(feature_classification, self.text_features_train)*self.scale
                # class_flow_logits = cos_sim(feature_classification_flow,  self.text_features_train)*self.scale_2

            else:
                class_text_logits = None

            if self.training:  # 通过真正标签找到对应的编码后的tensor，并且扩列【5，1，1024】 按照顺序找的
                context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)

            else:
                context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)
            # target_features = torch.cat([target_features,target_sub_feature.unsqueeze(1)],dim=1)
            # support_sub_feature = self.fc_sub(support_sub_feature)
            # support_sub_feature = self.flowTrans(support_sub_feature + context_support.squeeze(1),
            #                                      support_sub_feature + context_support.squeeze(1),
            #                                      support_sub_feature + context_support.squeeze(1))
            # target_sub_feature = self.fc_sub(target_sub_feature)
            # target_sub_feature = self.flowTrans(target_sub_feature, target_sub_feature, target_sub_feature)
            target_features = self.context2(target_features, target_features,target_features)  # 【5，8，1024】 支持集过transformer

            context_support = self.mid_layer(context_support)  # 【5，1，1024】 # 支持集对应的文字标签
            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                unique_labels = torch.unique(support_labels)
                support_features = [
                    torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels]
                support_features = torch.stack(support_features)
                context_support = [
                    torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels]
                context_support = torch.stack(context_support)
                if OPTICAL == True:
                    support_sub_feature =  [
                        torch.mean(torch.index_select(support_sub_feature, 0, extract_class_indices(support_labels, c)), dim=0)
                        for c in unique_labels]
                    support_sub_feature = torch.stack(support_sub_feature)
            support_features_fusion = torch.cat([support_features, context_support],dim=1)# stack操作 【5，9，1024】=【5，8，1024】+【5，1，1024】
            context_support_change = context_support.expand(-1, 8, -1)
            # support_features = torch.cat([support_features, context_support], dim=1)  # stack操作 【5，9，1024】
            # support_features = self.context2(support_features, support_features, support_features)[:,:self.args.DATA.NUM_INPUT_FRAMES, :]
            support_features = self.context2(support_features+context_support_change, support_features_fusion, support_features_fusion)[:,:self.args.DATA.NUM_INPUT_FRAMES, :]

            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                pass
            else:
                unique_labels = torch.unique(support_labels)
                support_features = [
                    torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels]
                support_features = torch.stack(support_features)  # [5,8,1024]
            # 对于每个唯一标签c，代码使用extract_class_indices函数从support_labels中提取所有标签为c的样本的索引。然后，使用torch.index_select函数从support_features中选取这些索引对应的特征向量，
            # 并使用torch.mean函数对选取的特征向量进行平均操作，得到一个代表该标签的平均特征向量。支持样本的特征向量通过这种方式被聚合成了一个由唯一标签对应的平均特征向量组成的列表(support_features)

            unique_labels = torch.unique(support_labels)

            n_queries = target_features.shape[0]
            n_support = support_features.shape[0]

            support_features = rearrange(support_features, 'b s d -> (b s) d')  # [40,1024]
            target_features = rearrange(target_features, 'b s d -> (b s) d')  # [200,1024]

            frame_sim = cos_sim(target_features, support_features)  # [200,40]
            if OPTICAL == True:
                flow_sim = cos_sim(target_sub_feature,support_sub_feature) #[25,5]
            frame_dists = 1 - frame_sim

            dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)  # [25, 5, 8, 8]

            # calculate query -> support and support -> query
            if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                cum_dists = OTAM_cum_dist_v2(dists)
            else:
                cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))

        else:
            if hasattr(self.args.TRAIN, "EVAL_TEXT") and self.args.TRAIN.EVAL_TEXT:
                unique_labels = torch.unique(support_labels)
                support_features, target_features, text_features = self.get_feats(support_images, target_images,
                                                                                  support_real_class)
                text_features = [
                    torch.mean(torch.index_select(text_features, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels]
                text_features = torch.stack(text_features)
                # unique_labels = torch.unique(support_labels)
                image_features = self.classification_layer(target_features.mean(1))
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = self.scale  # 1. # self.backbone.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                logits_per_image = F.softmax(logits_per_image, dim=1)

                cum_dists = -logits_per_image  #
                class_text_logits = None


            elif hasattr(self.args.TRAIN, "COMBINE") and self.args.TRAIN.COMBINE:
                # text_features = self.text_features_test[support_real_class.long()]
                unique_labels = torch.unique(support_labels)
                support_features, target_features, text_features = self.get_feats(support_images, target_images,
                                                                                  support_real_class)
                text_features = [
                    torch.mean(torch.index_select(text_features, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels]
                text_features = torch.stack(text_features)
                # unique_labels = torch.unique(support_labels)
                image_features = self.classification_layer(target_features.mean(1))
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = self.scale  # 1. # self.backbone.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                logits_per_image = F.softmax(logits_per_image, dim=1)

                class_text_logits = None

                support_bs = support_features.shape[0]
                target_bs = target_features.shape[0]

                feature_classification_in = torch.cat([support_features, target_features], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.text_features_train) * self.scale

                if self.training:
                    context_support = self.text_features_train[support_real_class.long()].unsqueeze(
                        1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)

                else:
                    context_support = self.text_features_test[support_real_class.long()].unsqueeze(
                        1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)

                target_features = self.context2(target_features, target_features, target_features)
                context_support = self.mid_layer(context_support)  # F.relu(self.mid_layer(context_support))
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    unique_labels = torch.unique(support_labels)
                    support_features = [
                        torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)),
                                   dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)
                    context_support = [
                        torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)),
                                   dim=0) for c in unique_labels]
                    context_support = torch.stack(context_support)
                support_features = torch.cat([support_features, context_support], dim=1)
                support_features = self.context2(support_features, support_features, support_features)[:,:self.args.DATA.NUM_INPUT_FRAMES, :]
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    pass
                else:
                    unique_labels = torch.unique(support_labels)
                    support_features = [
                        torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)),
                                   dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)

                unique_labels = torch.unique(support_labels)

                n_queries = target_features.shape[0]
                n_support = support_features.shape[0]

                support_features = rearrange(support_features, 'b s d -> (b s) d')
                target_features = rearrange(target_features, 'b s d -> (b s) d')

                frame_sim = cos_sim(target_features, support_features)
                frame_dists = 1 - frame_sim

                dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries,
                                  sb=n_support)  # [25, 25, 8, 8]

                # calculate query -> support and support -> query
                if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                    cum_dists_visual = OTAM_cum_dist_v2(dists)
                else:
                    cum_dists_visual = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(
                        rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
                cum_dists_visual_soft = F.softmax((8 - cum_dists_visual) / 8., dim=1)
                if hasattr(self.args.TRAIN, "TEXT_COFF") and self.args.TRAIN.TEXT_COFF:
                    cum_dists = -(logits_per_image.pow(self.args.TRAIN.TEXT_COFF) * cum_dists_visual_soft.pow(
                        1.0 - self.args.TRAIN.TEXT_COFF))
                else:
                    cum_dists = -(logits_per_image.pow(0.9) * cum_dists_visual_soft.pow(0.1))

                class_text_logits = None

            else:
                support_features, target_features, _ = self.get_feats(support_images, target_images, support_labels)
                support_bs = support_features.shape[0]
                target_bs = target_features.shape[0]

                # feature_classification_flow = torch.cat([support_sub_feature, target_sub_feature],dim=0) # [30]
                feature_classification_in = torch.cat([support_features, target_features], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                text_class = self.text_features_test[batch_class.long()]
                class_text_logits = cos_sim(feature_classification, text_class) * self.scale
                # class_flow_logits = cos_sim(feature_classification_flow, text_class) * self.scale_2



                if self.training:
                    context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)

                else:
                    context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)

                # target_features = torch.cat([target_features, target_sub_feature.unsqueeze(1)], dim=1)
                if OPTICAL == True:
                    with torch.no_grad():
                        support_sub_feature = self.opticalFlow_branch(support_sub_frame.permute(0, 2, 1, 3, 4))
                        target_sub_feature = self.opticalFlow_branch(target_sub_frame.permute(0, 2, 1, 3, 4))
                    support_sub_feature = self.fc_sub(support_sub_feature)
                    support_sub_feature = self.flowTrans(support_sub_feature, support_sub_feature, support_sub_feature)
                    target_sub_feature = self.fc_sub(target_sub_feature)
                    target_sub_feature = self.flowTrans(target_sub_feature, target_sub_feature, target_sub_feature)
                target_features = self.context2(target_features, target_features, target_features)
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    unique_labels = torch.unique(support_labels)
                    support_features = [
                        torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)),
                                   dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)
                    context_support = [
                        torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)),
                                   dim=0) for c in unique_labels]
                    context_support = torch.stack(context_support)
                    if OPTICAL == True:
                        support_sub_feature = [
                            torch.mean(torch.index_select(support_sub_feature, 0, extract_class_indices(support_labels, c)),
                                       dim=0) for c in unique_labels]
                        support_sub_feature = torch.stack(support_sub_feature)

                # support_features = torch.cat([support_features, context_support], dim=1)
                # support_features = self.context2(support_features, support_features, support_features)[:,:self.args.DATA.NUM_INPUT_FRAMES, :]
                support_features_fusion = torch.cat([support_features, context_support],dim=1)  # stack操作 【5，9，1024】=【5，8，1024】+【5，1，1024】
                context_support_change = context_support.expand(-1, 8, -1)
                support_features = self.context2(support_features + context_support_change, support_features_fusion,support_features_fusion)[:, :self.args.DATA.NUM_INPUT_FRAMES, :]
                # support_features = torch.cat([support_features, context_support], dim=1)
                # support_features = self.context2(support_features, support_features, support_features)[:,:self.args.DATA.NUM_INPUT_FRAMES, :]
                # support_features = (support_features + self.context2(support_features, support_features,support_features))[:,:self.args.DATA.NUM_INPUT_FRAMES,:]  # 【5，8，1024】通过这个切片操作，可以只保留support_features中前self.args.DATA.NUM_INPUT_FRAMES帧的特征，截取的结果将作为最终的support_features
                # target_features = self.context2(target_features, target_features, target_features)
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    pass
                else:
                    unique_labels = torch.unique(support_labels)
                    support_features = [
                        torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)),
                                   dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)

                unique_labels = torch.unique(support_labels)

                n_queries = target_features.shape[0]
                n_support = support_features.shape[0]

                support_features = rearrange(support_features, 'b s d -> (b s) d')  # [200, 2048]
                target_features = rearrange(target_features, 'b s d -> (b s) d')  # [200, 2048]

                frame_sim = cos_sim(target_features, support_features)  # [200, 200]
                if OPTICAL == True:
                    flow_sim = cos_sim(target_sub_feature, support_sub_feature)  # [25,5]
                frame_dists = 1 - frame_sim

                dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries,
                                  sb=n_support)  # [25, 25, 8, 8]

                # calculate query -> support and support -> query
                if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                    cum_dists = OTAM_cum_dist_v2(dists)
                else:
                    cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
        if OPTICAL == True:
            class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]  # list[5]
            # flow_sim = [torch.mean(torch.index_select(flow_sim, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
            class_dists = torch.stack(class_dists)  # [5,5]
            # flow_sim = torch.stack(flow_sim)
            class_dists = rearrange(class_dists, 'c q -> q c')
            # flow_sim = rearrange(flow_sim,'c q -> q c')
            return_dict = {'logits': - class_dists, "class_logits": class_text_logits,"class_flow_logits":flow_sim }#"temp_logits":temporal_text_logits
        else:
            class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]  # list[5]
            class_dists = torch.stack(class_dists)  # [5,5]
            class_dists = rearrange(class_dists, 'c q -> q c')
            return_dict = {'logits': - class_dists, "class_logits": class_text_logits,"class_flow_logits":False}#"temp_logits":temporal_text_logits


        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
# 光流融合策略（a）
visual_support = []
@HEAD_REGISTRY.register()
class CLIP_ADAPTER_OP(CNN_FSHead):
    def __init__(self, cfg):
        super(CLIP_ADAPTER_OP, self).__init__(cfg) # 进入到CNN_FSHead的初始化里面
        args = cfg
        self.args = cfg
        self.embedding = torch.nn.Embedding(77, 512)
        nn.init.normal_(self.embedding.weight, std=0.01)
        if cfg.VIDEO.HEAD.BACKBONE_NAME == "ViT-B/16":
            backbone, self.preprocess = load(cfg.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=cfg, jit=False)
            self.text = backbone.encode_text  # ViT-B/16
            self.backbone = backbone.visual  # model.load_state_dict(state_dict)
            self.class_real_train = cfg.TRAIN.CLASS_NAME
            self.class_real_test = cfg.TEST.CLASS_NAME
            self.mid_dim = 512
            self.opticalFlow_branch = InceptionI3d_load(device="cuda")
            self.fc_sub = nn.Linear(1024, self.mid_dim)
            nn.init.constant_(self.fc_sub.bias, 0.)
        else:
            print("error")
        with torch.no_grad():
            if AddPrompt == True:

                text_templete = [(self.class_real_train[int(ii)]) for ii in range(len(self.class_real_train))]
                self.text_templete_train = tokenize(text_templete).cuda()  # 把64个class_name全部变成tensor[64,77]
                self.text_embedding_train = backbone.encode_text_light(self.text_templete_train)  # [64,1024]
                text_templete = [(self.class_real_test[int(ii)]) for ii in range(len(self.class_real_test))]
                self.text_templete_test = tokenize(text_templete).cuda()
                self.text_embedding_test = backbone.encode_text_light(self.text_templete_test)
                self.actiondict_train = OrderedDict((self.class_real_train[i], self.text_embedding_train[i].cuda().cpu().data.numpy()) for i in range(len(self.class_real_train)))
                self.actiondict_test = OrderedDict(
                    (self.class_real_test[i], self.text_embedding_test[i].cuda().cpu().data.numpy()) for i in
                    range(len(self.class_real_test)))
                self.actiontoken_train = OrderedDict(
                    (self.class_real_train[i], self.text_templete_train[i]) for i in range(len(self.class_real_train)))
                self.actiontoken_test = OrderedDict(
                    (self.class_real_test[i], self.text_templete_test[i]) for i in range(len(self.class_real_test)))
            else:
                text_templete = ["a photo of {}".format(self.class_real_train[int(ii)]) for ii in range(len(self.class_real_train))]
                text_templete = tokenize(text_templete).cuda()  # 把64个句子全部变成token
                self.text_features_train = backbone.encode_text(text_templete)

                text_templete = ["a photo of {}".format(self.class_real_test[int(ii)]) for ii in range(len(self.class_real_test))]
                text_templete = tokenize(text_templete).cuda()

                self.text_features_test = backbone.encode_text(text_templete)

        # self.decoder = GRU()
        self.mid_layer = nn.Sequential()  # 定义中间层
        self.classification_layer = nn.Sequential()  # 定义分类层
        self.scale = nn.Parameter(torch.FloatTensor(1),requires_grad=True)  # 定义一个学习参数learnable parameter，它是一个浮点数的张量（torch.FloatTensor）。
        self.scale.data.fill_(1.0)
        # self.scale_2 = nn.Parameter(torch.FloatTensor(1),requires_grad=True)  # 定义一个学习参数learnable parameter，它是一个浮点数的张量（torch.FloatTensor）。
        # self.scale_2.data.fill_(1.0)
        # nn.Parameter 函数用于将张量转换为可学习的参数。这个参数的初始值被填充为 1.0

        if hasattr(self.args.TRAIN, "TRANSFORMER_DEPTH") and self.args.TRAIN.TRANSFORMER_DEPTH:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2,
                                           depth=int(self.args.TRAIN.TRANSFORMER_DEPTH))
        else:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2)
        # self.flowTrans  = Transformer_flow(width=self.mid_dim, layers=1, heads=8, attn_mask=None)
        self.flowTrans = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2)
        # self.mha = MultiheadAttention(embed_dim=self.mid_dim, num_heads=8, dropout=0.2)
    def process_prompt(self, class_name, actiondict, actiontoken, prefix=16, postfix=16):
        self.text_embedding = self.embedding(torch.arange(77).cuda())[None, :].repeat([len(class_name), 1, 1]).to(
            "cuda")  # [64,77,512]
        self.prompt_actiontoken = torch.zeros(len(class_name), 77)

        for i, a in enumerate(class_name):
            emb = torch.from_numpy(actiondict[a]).float().to("cuda")  # [77,512]
            token = torch.from_numpy(np.array(actiontoken[a].cpu().numpy()))  # [77]
            self.text_embedding[i][0] = emb[0]  # [512]
            ind = np.argmax(token, -1)

            self.text_embedding[i][prefix + 1:prefix + ind] = emb[1:ind]  # [17:19]=[1,3]
            self.text_embedding[i][prefix + ind + postfix] = emb[ind]  # [33]=[3]

            self.prompt_actiontoken[i][0] = token[0]  # []
            self.prompt_actiontoken[i][prefix + 1:prefix + ind] = token[1:ind]  # [17:19]=[1:3]
            self.prompt_actiontoken[i][prefix + ind + postfix] = token[ind]

        self.text_embedding.to("cuda")
        self.prompt_actiontoken.to("cuda")

    def get_feats(self, support_images, target_images, support_real_class=False, support_labels=False):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        if self.training:

            support_features = self.backbone(support_images.permute(0,2,1,3,4)).squeeze()
            # os.system("nvidia-smi")
            target_features = self.backbone(target_images.permute(0,2,1,3,4)).squeeze() #[5,8,1024]

            dim = int(support_features.shape[1])  # [2048]


            support_features_text = None

        else:
            support_features = self.backbone(support_images.permute(0,2,1,3,4)).squeeze()
            target_features = self.backbone(target_images.permute(0,2,1,3,4)).squeeze()

            support_features_text = self.text_features_test[support_real_class.long()]

        return support_features, target_features, support_features_text

    def forward(self, inputs):
        support_images, support_labels, target_images, support_real_class,batch_class ,support_sub_frame,target_sub_frame= inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['real_support_labels'],inputs['batch_class_list'],inputs['support_subframe_set'],inputs['target_subframe_set'] # [200, 3, 224, 224] inputs["real_support_labels"]
        # set_trace()
        if AddPrompt == True:

            self.process_prompt(self.class_real_train, self.actiondict_train, self.actiontoken_train)
            self.text_features_train = self.text(self.text_embedding, self.prompt_actiontoken)
            self.process_prompt(self.class_real_test, self.actiondict_test, self.actiontoken_test)
            self.text_features_test = self.text(self.text_embedding, self.prompt_actiontoken)
        else:
            pass
        if self.training:
            support_features, target_features, _ = self.get_feats(support_images, target_images,support_labels)

            if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
                feature_classification_in = torch.cat([support_features, target_features],dim=0)  # 在0维度上进行cat【30，8，512】
                # feature_classification_flow = torch.cat([support_sub_feature, target_sub_feature],dim=0) # [30]
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.text_features_train)*self.scale
                # class_flow_logits = cos_sim(feature_classification_flow,  self.text_features_train)*self.scale_2
            else:
                class_text_logits = None

            if self.training:  # 通过真正标签找到对应的编码后的tensor，并且扩列【5，1，1024】 按照顺序找的
                context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)

            else:
                context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)
            if OPTICAL == True:# [5,8,1024]
                with torch.no_grad():
                    support_sub_feature = self.opticalFlow_branch(support_sub_frame.permute(0,2,1,3,4))
                    target_sub_feature = self.opticalFlow_branch(target_sub_frame.permute(0,2,1,3,4))
                support_sub_feature = self.fc_sub(support_sub_feature) # 1024 -> 512
                support_sub_feature = support_sub_feature.unsqueeze(1) * support_features # 光流扩维 -> 与支持集合相乘
                support_sub_feature= self.flowTrans(support_sub_feature,support_sub_feature, support_sub_feature) # 一同进入tansformer
                target_sub_feature = self.fc_sub(target_sub_feature)  # 查询集也做相同动作
                target_sub_feature = target_sub_feature.unsqueeze(1) * target_features
                target_sub_feature=  self.flowTrans(target_sub_feature,target_sub_feature,target_sub_feature)
            # target_features = torch.cat([target_features,target_sub_feature.unsqueeze(1)],dim=1)
            # support_sub_feature = self.fc_sub(support_sub_feature)
            # support_sub_feature = self.flowTrans(support_sub_feature + context_support.squeeze(1),
            #                                      support_sub_feature + context_support.squeeze(1),
            #                                      support_sub_feature + context_support.squeeze(1))
            # target_sub_feature = self.fc_sub(target_sub_feature)
            # target_sub_feature = self.flowTrans(target_sub_feature, target_sub_feature, target_sub_feature)
            target_features = self.context2(target_features, target_features,target_features)  # 【5，8，1024】 支持集过transformer

            context_support = self.mid_layer(context_support)  # 【5，1，1024】 # 支持集对应的文字标签
            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                unique_labels = torch.unique(support_labels)
                support_features = [
                    torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels]
                support_features = torch.stack(support_features)
                context_support = [
                    torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels]
                context_support = torch.stack(context_support)
                if OPTICAL == True:
                    support_sub_feature =  [
                        torch.mean(torch.index_select(support_sub_feature, 0, extract_class_indices(support_labels, c)), dim=0)
                        for c in unique_labels]
                    support_sub_feature = torch.stack(support_sub_feature)
            support_features_fusion = torch.cat([support_features, context_support],dim=1)# stack操作 【5，9，1024】=【5，8，1024】+【5，1，1024】
            context_support_change = context_support.expand(-1, 8, -1)
            # support_features = torch.cat([support_features, context_support], dim=1)  # stack操作 【5，9，1024】
            # support_features = self.context2(support_features, support_features, support_features)[:,:self.args.DATA.NUM_INPUT_FRAMES, :]
            support_features = self.context2(support_features+context_support_change, support_features_fusion, support_features_fusion)[:,:self.args.DATA.NUM_INPUT_FRAMES, :]

            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                pass
            else:
                unique_labels = torch.unique(support_labels)
                support_features = [
                    torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels]
                support_features = torch.stack(support_features)  # [5,8,1024]
            # 对于每个唯一标签c，代码使用extract_class_indices函数从support_labels中提取所有标签为c的样本的索引。然后，使用torch.index_select函数从support_features中选取这些索引对应的特征向量，
            # 并使用torch.mean函数对选取的特征向量进行平均操作，得到一个代表该标签的平均特征向量。支持样本的特征向量通过这种方式被聚合成了一个由唯一标签对应的平均特征向量组成的列表(support_features)

            unique_labels = torch.unique(support_labels)

            n_queries = target_features.shape[0]
            n_support = support_features.shape[0]

            support_features = rearrange(support_features, 'b s d -> (b s) d')  # [40,1024]
            support_sub_feature = rearrange(support_sub_feature, 'b s d -> (b s) d')  # [40,1024]
            target_features = rearrange(target_features, 'b s d -> (b s) d')  # [200,1024]
            target_sub_feature = rearrange(target_sub_feature, 'b s d -> (b s) d')  # [200,1024]

            frame_sim = cos_sim(target_features, support_features)  # [200,40]
            if OPTICAL == True:
                flow_sim = cos_sim(target_sub_feature,support_sub_feature) #[25,5]
            frame_dists = 1 - frame_sim
            flow_dists = 1 - flow_sim
            dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)  # [25, 5, 8, 8]
            flow_dists = rearrange(flow_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)  # [25, 5, 8, 8]

            # calculate query -> support and support -> query

            cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
            cum_flow = OTAM_cum_dist_v2(flow_dists) + OTAM_cum_dist_v2(rearrange(flow_dists, 'tb sb ts ss -> tb sb ss ts'))
        else:
            support_features, target_features, _ = self.get_feats(support_images, target_images, support_labels)
            support_bs = support_features.shape[0]
            target_bs = target_features.shape[0]

            feature_classification_in = torch.cat([support_features, target_features], dim=0)
            feature_classification = self.classification_layer(feature_classification_in).mean(1)
            text_class = self.text_features_test[batch_class.long()]
            class_text_logits = cos_sim(feature_classification, text_class) * self.scale




            if self.training:
                context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)

            else:
                context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)

            if OPTICAL == True:  # [5,8,1024]
                with torch.no_grad():
                    support_sub_feature = self.opticalFlow_branch(support_sub_frame.permute(0, 2, 1, 3, 4))
                    target_sub_feature = self.opticalFlow_branch(target_sub_frame.permute(0, 2, 1, 3, 4))
                support_sub_feature = self.fc_sub(support_sub_feature)
                support_sub_feature = support_sub_feature.unsqueeze(1) * support_features
                support_sub_feature = self.flowTrans(support_sub_feature, support_sub_feature, support_sub_feature)
                target_sub_feature = self.fc_sub(target_sub_feature)
                target_sub_feature = target_sub_feature.unsqueeze(1) * target_features
                target_sub_feature = self.flowTrans(target_sub_feature, target_sub_feature, target_sub_feature)
            target_features = self.context2(target_features, target_features, target_features)
            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                unique_labels = torch.unique(support_labels)
                support_features = [
                    torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)),
                               dim=0) for c in unique_labels]
                support_features = torch.stack(support_features)
                context_support = [
                    torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)),
                               dim=0) for c in unique_labels]
                context_support = torch.stack(context_support)
                if OPTICAL == True:
                    support_sub_feature = [
                        torch.mean(torch.index_select(support_sub_feature, 0, extract_class_indices(support_labels, c)),
                                   dim=0) for c in unique_labels]
                    support_sub_feature = torch.stack(support_sub_feature)
            support_features_fusion = torch.cat([support_features, context_support],dim=1)  # stack操作 【5，9，1024】=【5，8，1024】+【5，1，1024】
            context_support_change = context_support.expand(-1, 8, -1)
            support_features = self.context2(support_features + context_support_change, support_features_fusion,support_features_fusion)[:, :self.args.DATA.NUM_INPUT_FRAMES, :]
            # torch.save(support_features.mean(1),"/home/imi1214/data1/xiaqing/Vit_adapter_clip_FSAR/visual/s1.pt")
            # visual_support.append(support_features.mean(1))
            # if len(visual_support) == 100:
            #     print("finish")
            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                pass
            else:
                unique_labels = torch.unique(support_labels)
                support_features = [
                    torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)),
                               dim=0) for c in unique_labels]
                support_features = torch.stack(support_features)

            unique_labels = torch.unique(support_labels)

            n_queries = target_features.shape[0]
            n_support = support_features.shape[0]

            support_features = rearrange(support_features, 'b s d -> (b s) d')  # [40,1024]
            support_sub_feature = rearrange(support_sub_feature, 'b s d -> (b s) d')  # [40,1024]
            target_features = rearrange(target_features, 'b s d -> (b s) d')  # [200,1024]
            target_sub_feature = rearrange(target_sub_feature, 'b s d -> (b s) d')  # [200,1024]

            frame_sim = cos_sim(target_features, support_features)  # [200,40]
            if OPTICAL == True:
                flow_sim = cos_sim(target_sub_feature, support_sub_feature)  # [25,5]
            frame_dists = 1 - frame_sim
            flow_dists = 1 - flow_sim
            dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries,sb=n_support)  # [25, 5, 8, 8]
            flow_dists = rearrange(flow_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries,sb=n_support)  # [25, 5, 8, 8]

            cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
            cum_flow = OTAM_cum_dist_v2(flow_dists) + OTAM_cum_dist_v2(rearrange(flow_dists, 'tb sb ts ss -> tb sb ss ts'))
        if OPTICAL == True:
            class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]  # list[5]
            flow_sim = [torch.mean(torch.index_select(cum_flow, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
            class_dists = torch.stack(class_dists)  # [5,5]
            flow_sim = torch.stack(flow_sim)
            class_dists = rearrange(class_dists, 'c q -> q c')
            flow_sim = rearrange(flow_sim,'c q -> q c')
            return_dict = {'logits': - class_dists, "class_logits": class_text_logits,"class_flow_logits": - flow_sim }#"temp_logits":temporal_text_logits
        else:
            class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]  # list[5]
            class_dists = torch.stack(class_dists)  # [5,5]
            class_dists = rearrange(class_dists, 'c q -> q c')
            return_dict = {'logits': - class_dists, "class_logits": class_text_logits,"class_flow_logits":False}#"temp_logits":temporal_text_logits


        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
# 光流融合策略（b）
@HEAD_REGISTRY.register()
class CLIP_ADAPTER_VT_MV(CNN_FSHead):
    def __init__(self, cfg):
        super(CLIP_ADAPTER_VT_MV, self).__init__(cfg) # 进入到CNN_FSHead的初始化里面
        args = cfg
        self.args = cfg
        self.embedding = torch.nn.Embedding(77, 512)
        nn.init.normal_(self.embedding.weight, std=0.01)
        if cfg.VIDEO.HEAD.BACKBONE_NAME == "ViT-B/16":
            backbone, self.preprocess = load(cfg.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=cfg, jit=False)
            self.text = backbone.encode_text  # ViT-B/16
            self.backbone = backbone.visual  # model.load_state_dict(state_dict)
            self.class_real_train = cfg.TRAIN.CLASS_NAME
            self.class_real_test = cfg.TEST.CLASS_NAME
            self.mid_dim = 512
            self.opticalFlow_branch = InceptionI3d_load(device="cuda")
            self.fc_sub = nn.Linear(1024, self.mid_dim)
            nn.init.constant_(self.fc_sub.bias, 0.)


        else:
            print("error")
        with torch.no_grad():
            if AddPrompt == True:

                text_templete = [(self.class_real_train[int(ii)]) for ii in range(len(self.class_real_train))]
                self.text_templete_train = tokenize(text_templete).cuda()  # 把64个class_name全部变成tensor[64,77]
                self.text_embedding_train = backbone.encode_text_light(self.text_templete_train)  # [64,1024]
                text_templete = [(self.class_real_test[int(ii)]) for ii in range(len(self.class_real_test))]
                self.text_templete_test = tokenize(text_templete).cuda()
                self.text_embedding_test = backbone.encode_text_light(self.text_templete_test)
                self.actiondict_train = OrderedDict((self.class_real_train[i], self.text_embedding_train[i].cuda().cpu().data.numpy()) for i in range(len(self.class_real_train)))
                self.actiondict_test = OrderedDict(
                    (self.class_real_test[i], self.text_embedding_test[i].cuda().cpu().data.numpy()) for i in
                    range(len(self.class_real_test)))
                self.actiontoken_train = OrderedDict(
                    (self.class_real_train[i], self.text_templete_train[i]) for i in range(len(self.class_real_train)))
                self.actiontoken_test = OrderedDict(
                    (self.class_real_test[i], self.text_templete_test[i]) for i in range(len(self.class_real_test)))
            else:
                text_templete = ["a photo of {}".format(self.class_real_train[int(ii)]) for ii in range(len(self.class_real_train))]
                text_templete = tokenize(text_templete).cuda()  # 把64个句子全部变成token
                self.text_features_train = backbone.encode_text(text_templete)

                text_templete = ["a photo of {}".format(self.class_real_test[int(ii)]) for ii in range(len(self.class_real_test))]
                text_templete = tokenize(text_templete).cuda()

                self.text_features_test = backbone.encode_text(text_templete)

        # self.decoder = GRU()
        self.mid_layer = nn.Sequential()  # 定义中间层
        self.classification_layer = nn.Sequential()  # 定义分类层
        self.scale = nn.Parameter(torch.FloatTensor(1),requires_grad=True)  # 定义一个学习参数learnable parameter，它是一个浮点数的张量（torch.FloatTensor）。
        self.scale.data.fill_(1.0)


        if hasattr(self.args.TRAIN, "TRANSFORMER_DEPTH") and self.args.TRAIN.TRANSFORMER_DEPTH:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2,
                                           depth=int(self.args.TRAIN.TRANSFORMER_DEPTH))
        else:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2)

        self.flowTrans = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2)

    def process_prompt(self, class_name, actiondict, actiontoken, prefix=16, postfix=16):
        self.text_embedding = self.embedding(torch.arange(77).cuda())[None, :].repeat([len(class_name), 1, 1]).to(
            "cuda")  # [64,77,512]
        self.prompt_actiontoken = torch.zeros(len(class_name), 77)

        for i, a in enumerate(class_name):
            emb = torch.from_numpy(actiondict[a]).float().to("cuda")  # [77,512]
            token = torch.from_numpy(np.array(actiontoken[a].cpu().numpy()))  # [77]
            self.text_embedding[i][0] = emb[0]  # [512]
            ind = np.argmax(token, -1)

            self.text_embedding[i][prefix + 1:prefix + ind] = emb[1:ind]  # [17:19]=[1,3]
            self.text_embedding[i][prefix + ind + postfix] = emb[ind]  # [33]=[3]

            self.prompt_actiontoken[i][0] = token[0]  # []
            self.prompt_actiontoken[i][prefix + 1:prefix + ind] = token[1:ind]  # [17:19]=[1:3]
            self.prompt_actiontoken[i][prefix + ind + postfix] = token[ind]

        self.text_embedding.to("cuda")
        self.prompt_actiontoken.to("cuda")

    def get_feats(self, support_images, target_images, support_real_class=False, support_labels=False):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        if self.training:

            support_features = self.backbone(support_images.permute(0,2,1,3,4)).squeeze()
            # os.system("nvidia-smi")
            target_features = self.backbone(target_images.permute(0,2,1,3,4)).squeeze() #[5,8,1024]

            dim = int(support_features.shape[1])  # [2048]


            support_features_text = None

        else:
            support_features = self.backbone(support_images.permute(0,2,1,3,4)).squeeze()
            target_features = self.backbone(target_images.permute(0,2,1,3,4)).squeeze()

            support_features_text = self.text_features_test[support_real_class.long()]

        return support_features, target_features, support_features_text

    def forward(self, inputs):
        support_images, support_labels, target_images, support_real_class,batch_class = inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['real_support_labels'],inputs['batch_class_list']  # [200, 3, 224, 224] inputs["real_support_labels"]
        support_sub_frame,target_sub_frame = inputs['support_subframe_set'],inputs['target_subframe_set']
        # set_trace()
        if AddPrompt == True:

            self.process_prompt(self.class_real_train, self.actiondict_train, self.actiontoken_train)
            self.text_features_train = self.text(self.text_embedding, self.prompt_actiontoken)
            self.process_prompt(self.class_real_test, self.actiondict_test, self.actiontoken_test)
            self.text_features_test = self.text(self.text_embedding, self.prompt_actiontoken)
        else:
            pass
        if self.training:
            support_features, target_features, _ = self.get_feats(support_images, target_images,support_labels)


            if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
                feature_classification_in = torch.cat([support_features, target_features],dim=0)  # 在0维度上进行cat【30，8，512】
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.text_features_train)*self.scale

            else:
                class_text_logits = None

            if self.training:  # 通过真正标签找到对应的编码后的tensor，并且扩列【5，1，1024】 按照顺序找的
                context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)
            else:
                context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)

            target_features = self.context2(target_features, target_features,target_features)  # 【5，8，1024】 支持集过transformer
            context_support = self.mid_layer(context_support)  # 【5，1，1024】 # 支持集对应的文字标签
            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                unique_labels = torch.unique(support_labels)
                support_features = [
                    torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels]
                support_features = torch.stack(support_features)
                context_support = [
                    torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels]
                context_support = torch.stack(context_support)
                # if OPTICAL == True:
                #     support_sub_feature =  [
                #         torch.mean(torch.index_select(support_sub_feature, 0, extract_class_indices(support_labels, c)), dim=0)
                #         for c in unique_labels]
                #     support_sub_feature = torch.stack(support_sub_feature)
            support_features_fusion = torch.cat([support_features, context_support],dim=1)# stack操作 【5，9，1024】=【5，8，1024】+【5，1，1024】
            context_support_change = context_support.expand(-1, 8, -1)
            support_features = self.context2(support_features+context_support_change, support_features_fusion, support_features_fusion)[:,:self.args.DATA.NUM_INPUT_FRAMES, :]
            if OPTICAL == True:# [5,8,1024]
                with torch.no_grad():
                    support_sub_feature = self.opticalFlow_branch(support_sub_frame.permute(0,2,1,3,4))
                    target_sub_feature = self.opticalFlow_branch(target_sub_frame.permute(0,2,1,3,4))
                support_sub_feature = self.fc_sub(support_sub_feature)
                support_sub_feature = support_sub_feature.unsqueeze(1) * support_features
                support_sub_feature= self.flowTrans(support_sub_feature,support_sub_feature, support_sub_feature)
                target_sub_feature = self.fc_sub(target_sub_feature)
                target_sub_feature = target_sub_feature.unsqueeze(1) * target_features
                target_sub_feature=  self.flowTrans(target_sub_feature,target_sub_feature,target_sub_feature)

            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                pass
            else:
                unique_labels = torch.unique(support_labels)
                support_features = [
                    torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels]
                support_features = torch.stack(support_features)  # [5,8,1024]
            # 对于每个唯一标签c，代码使用extract_class_indices函数从support_labels中提取所有标签为c的样本的索引。然后，使用torch.index_select函数从support_features中选取这些索引对应的特征向量，
            # 并使用torch.mean函数对选取的特征向量进行平均操作，得到一个代表该标签的平均特征向量。支持样本的特征向量通过这种方式被聚合成了一个由唯一标签对应的平均特征向量组成的列表(support_features)

            unique_labels = torch.unique(support_labels)

            n_queries = target_features.shape[0]
            n_support = support_features.shape[0]

            support_features = rearrange(support_features, 'b s d -> (b s) d')  # [40,1024]
            support_sub_feature = rearrange(support_sub_feature, 'b s d -> (b s) d')  # [40,1024]
            target_features = rearrange(target_features, 'b s d -> (b s) d')  # [200,1024]
            target_sub_feature = rearrange(target_sub_feature, 'b s d -> (b s) d')  # [200,1024]

            frame_sim = cos_sim(target_features, support_features)  # [200,40]
            if OPTICAL == True:
                flow_sim = cos_sim(target_sub_feature,support_sub_feature) #[25,5]
            frame_dists = 1 - frame_sim
            flow_dists = 1 - flow_sim
            dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)  # [25, 5, 8, 8]
            flow_dists = rearrange(flow_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)  # [25, 5, 8, 8]

            # calculate query -> support and support -> query

            cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
            cum_flow = OTAM_cum_dist_v2(flow_dists) + OTAM_cum_dist_v2(rearrange(flow_dists, 'tb sb ts ss -> tb sb ss ts'))
        else:
            support_features, target_features, _ = self.get_feats(support_images, target_images, support_labels)
            support_bs = support_features.shape[0]
            target_bs = target_features.shape[0]

            feature_classification_in = torch.cat([support_features, target_features], dim=0)
            feature_classification = self.classification_layer(feature_classification_in).mean(1)
            text_class = self.text_features_test[batch_class.long()]
            class_text_logits = cos_sim(feature_classification, text_class) * self.scale




            if self.training:
                context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)

            else:
                context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)


            target_features = self.context2(target_features, target_features, target_features)
            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                unique_labels = torch.unique(support_labels)
                support_features = [
                    torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)),
                               dim=0) for c in unique_labels]
                support_features = torch.stack(support_features)
                context_support = [
                    torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)),
                               dim=0) for c in unique_labels]
                context_support = torch.stack(context_support)
                # if OPTICAL == True:
                    # support_sub_feature = [
                    #     torch.mean(torch.index_select(support_sub_feature, 0, extract_class_indices(support_labels, c)),
                    #                dim=0) for c in unique_labels]
                    # support_sub_feature = torch.stack(support_sub_feature)


            support_features_fusion = torch.cat([support_features, context_support],dim=1)  # stack操作 【5，9，1024】=【5，8，1024】+【5，1，1024】
            context_support_change = context_support.expand(-1, 8, -1)
            support_features = self.context2(support_features + context_support_change, support_features_fusion,support_features_fusion)[:, :self.args.DATA.NUM_INPUT_FRAMES, :]
            if OPTICAL == True:  # [5,8,1024]
                with torch.no_grad():
                    support_sub_feature = self.opticalFlow_branch(support_sub_frame.permute(0, 2, 1, 3, 4))
                    target_sub_feature = self.opticalFlow_branch(target_sub_frame.permute(0, 2, 1, 3, 4))
                support_sub_feature = self.fc_sub(support_sub_feature)
                support_sub_feature = support_sub_feature.unsqueeze(1) * support_features
                support_sub_feature = self.flowTrans(support_sub_feature, support_sub_feature, support_sub_feature)
                target_sub_feature = self.fc_sub(target_sub_feature)
                target_sub_feature = target_sub_feature.unsqueeze(1) * target_features
                target_sub_feature = self.flowTrans(target_sub_feature, target_sub_feature, target_sub_feature)
            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                pass
            else:
                unique_labels = torch.unique(support_labels)
                support_features = [
                    torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)),
                               dim=0) for c in unique_labels]
                support_features = torch.stack(support_features)

            unique_labels = torch.unique(support_labels)

            n_queries = target_features.shape[0]
            n_support = support_features.shape[0]

            support_features = rearrange(support_features, 'b s d -> (b s) d')  # [40,1024]
            support_sub_feature = rearrange(support_sub_feature, 'b s d -> (b s) d')  # [40,1024]
            target_features = rearrange(target_features, 'b s d -> (b s) d')  # [200,1024]
            target_sub_feature = rearrange(target_sub_feature, 'b s d -> (b s) d')  # [200,1024]

            frame_sim = cos_sim(target_features, support_features)  # [200,40]
            if OPTICAL == True:
                flow_sim = cos_sim(target_sub_feature, support_sub_feature)  # [25,5]
            frame_dists = 1 - frame_sim
            flow_dists = 1 - flow_sim
            dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries,sb=n_support)  # [25, 5, 8, 8]
            flow_dists = rearrange(flow_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries,sb=n_support)  # [25, 5, 8, 8]

            cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
            cum_flow = OTAM_cum_dist_v2(flow_dists) + OTAM_cum_dist_v2(rearrange(flow_dists, 'tb sb ts ss -> tb sb ss ts'))
        if OPTICAL == True:
            class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]  # list[5]
            flow_sim = [torch.mean(torch.index_select(cum_flow, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
            class_dists = torch.stack(class_dists)  # [5,5]
            flow_sim = torch.stack(flow_sim)
            class_dists = rearrange(class_dists, 'c q -> q c')
            flow_sim = rearrange(flow_sim,'c q -> q c')
            return_dict = {'logits': - class_dists, "class_logits": class_text_logits,"class_flow_logits": - flow_sim }#"temp_logits":temporal_text_logits
        else:
            class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]  # list[5]
            class_dists = torch.stack(class_dists)  # [5,5]
            class_dists = rearrange(class_dists, 'c q -> q c')
            return_dict = {'logits': - class_dists, "class_logits": class_text_logits,"class_flow_logits":False}#"temp_logits":temporal_text_logits


        return return_dict
# 光流融合策略（c）
@HEAD_REGISTRY.register()
class CLIP_ADAPTER_MV_VT(CNN_FSHead):
    def __init__(self, cfg):
        super(CLIP_ADAPTER_MV_VT, self).__init__(cfg) # 进入到CNN_FSHead的初始化里面
        args = cfg
        self.args = cfg
        self.embedding = torch.nn.Embedding(77, 512)
        nn.init.normal_(self.embedding.weight, std=0.01)
        if cfg.VIDEO.HEAD.BACKBONE_NAME == "ViT-B/16":
            backbone, self.preprocess = load(cfg.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=cfg, jit=False)
            self.text = backbone.encode_text  # ViT-B/16
            self.backbone = backbone.visual  # model.load_state_dict(state_dict)
            self.class_real_train = cfg.TRAIN.CLASS_NAME
            self.class_real_test = cfg.TEST.CLASS_NAME
            self.mid_dim = 512
            self.opticalFlow_branch = InceptionI3d_load(device="cuda")
            self.fc_sub = nn.Linear(1024, self.mid_dim)
            nn.init.constant_(self.fc_sub.bias, 0.)


        else:
            print("error")
        with torch.no_grad():
            if AddPrompt == True:

                text_templete = [(self.class_real_train[int(ii)]) for ii in range(len(self.class_real_train))]
                self.text_templete_train = tokenize(text_templete).cuda()  # 把64个class_name全部变成tensor[64,77]
                self.text_embedding_train = backbone.encode_text_light(self.text_templete_train)  # [64,1024]
                text_templete = [(self.class_real_test[int(ii)]) for ii in range(len(self.class_real_test))]
                self.text_templete_test = tokenize(text_templete).cuda()
                self.text_embedding_test = backbone.encode_text_light(self.text_templete_test)
                self.actiondict_train = OrderedDict((self.class_real_train[i], self.text_embedding_train[i].cuda().cpu().data.numpy()) for i in range(len(self.class_real_train)))
                self.actiondict_test = OrderedDict(
                    (self.class_real_test[i], self.text_embedding_test[i].cuda().cpu().data.numpy()) for i in
                    range(len(self.class_real_test)))
                self.actiontoken_train = OrderedDict(
                    (self.class_real_train[i], self.text_templete_train[i]) for i in range(len(self.class_real_train)))
                self.actiontoken_test = OrderedDict(
                    (self.class_real_test[i], self.text_templete_test[i]) for i in range(len(self.class_real_test)))
            else:
                text_templete = ["a photo of {}".format(self.class_real_train[int(ii)]) for ii in range(len(self.class_real_train))]
                text_templete = tokenize(text_templete).cuda()  # 把64个句子全部变成token
                self.text_features_train = backbone.encode_text(text_templete)

                text_templete = ["a photo of {}".format(self.class_real_test[int(ii)]) for ii in range(len(self.class_real_test))]
                text_templete = tokenize(text_templete).cuda()

                self.text_features_test = backbone.encode_text(text_templete)

        # self.decoder = GRU()
        self.mid_layer = nn.Sequential()  # 定义中间层
        self.classification_layer = nn.Sequential()  # 定义分类层
        self.scale = nn.Parameter(torch.FloatTensor(1),requires_grad=True)  # 定义一个学习参数learnable parameter，它是一个浮点数的张量（torch.FloatTensor）。
        self.scale.data.fill_(1.0)


        if hasattr(self.args.TRAIN, "TRANSFORMER_DEPTH") and self.args.TRAIN.TRANSFORMER_DEPTH:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2,
                                           depth=int(self.args.TRAIN.TRANSFORMER_DEPTH))
        else:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2)

        self.flowTrans = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2)

    def process_prompt(self, class_name, actiondict, actiontoken, prefix=16, postfix=16):
        self.text_embedding = self.embedding(torch.arange(77).cuda())[None, :].repeat([len(class_name), 1, 1]).to(
            "cuda")  # [64,77,512]
        self.prompt_actiontoken = torch.zeros(len(class_name), 77)

        for i, a in enumerate(class_name):
            emb = torch.from_numpy(actiondict[a]).float().to("cuda")  # [77,512]
            token = torch.from_numpy(np.array(actiontoken[a].cpu().numpy()))  # [77]
            self.text_embedding[i][0] = emb[0]  # [512]
            ind = np.argmax(token, -1)

            self.text_embedding[i][prefix + 1:prefix + ind] = emb[1:ind]  # [17:19]=[1,3]
            self.text_embedding[i][prefix + ind + postfix] = emb[ind]  # [33]=[3]

            self.prompt_actiontoken[i][0] = token[0]  # []
            self.prompt_actiontoken[i][prefix + 1:prefix + ind] = token[1:ind]  # [17:19]=[1:3]
            self.prompt_actiontoken[i][prefix + ind + postfix] = token[ind]

        self.text_embedding.to("cuda")
        self.prompt_actiontoken.to("cuda")

    def get_feats(self, support_images, target_images, support_real_class=False, support_labels=False):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        if self.training:

            support_features = self.backbone(support_images.permute(0,2,1,3,4)).squeeze()
            # os.system("nvidia-smi")
            target_features = self.backbone(target_images.permute(0,2,1,3,4)).squeeze() #[5,8,1024]

            dim = int(support_features.shape[1])  # [2048]


            support_features_text = None

        else:
            support_features = self.backbone(support_images.permute(0,2,1,3,4)).squeeze()
            target_features = self.backbone(target_images.permute(0,2,1,3,4)).squeeze()

            support_features_text = self.text_features_test[support_real_class.long()]

        return support_features, target_features, support_features_text

    def forward(self, inputs):
        support_images, support_labels, target_images, support_real_class,batch_class = inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['real_support_labels'],inputs['batch_class_list']  # [200, 3, 224, 224] inputs["real_support_labels"]
        support_sub_frame,target_sub_frame = inputs['support_subframe_set'],inputs['target_subframe_set']
        # set_trace()
        if AddPrompt == True:

            self.process_prompt(self.class_real_train, self.actiondict_train, self.actiontoken_train)
            self.text_features_train = self.text(self.text_embedding, self.prompt_actiontoken)
            self.process_prompt(self.class_real_test, self.actiondict_test, self.actiontoken_test)
            self.text_features_test = self.text(self.text_embedding, self.prompt_actiontoken)
        else:
            pass
        if self.training:
            support_features, target_features, _ = self.get_feats(support_images, target_images,support_labels)


            if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
                feature_classification_in = torch.cat([support_features, target_features],dim=0)  # 在0维度上进行cat【30，8，512】
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.text_features_train)*self.scale

            else:
                class_text_logits = None

            if self.training:  # 通过真正标签找到对应的编码后的tensor，并且扩列【5，1，1024】 按照顺序找的
                context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)
            else:
                context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)
            if OPTICAL == True:# [5,8,1024]
                with torch.no_grad():
                    support_sub_feature = self.opticalFlow_branch(support_sub_frame.permute(0,2,1,3,4))
                    target_sub_feature = self.opticalFlow_branch(target_sub_frame.permute(0,2,1,3,4))
                support_sub_feature = self.fc_sub(support_sub_feature)
                support_sub_feature = support_sub_feature.unsqueeze(1) * support_features
                support_sub_feature= self.flowTrans(support_sub_feature,support_sub_feature, support_sub_feature)
                target_sub_feature = self.fc_sub(target_sub_feature)
                target_sub_feature = target_sub_feature.unsqueeze(1) * target_features
                target_sub_feature=  self.flowTrans(target_sub_feature,target_sub_feature,target_sub_feature)

            target_features = self.context2(target_features, target_features,target_features)  # 【5，8，1024】 支持集过transformer
            context_support = self.mid_layer(context_support)  # 【5，1，1024】 # 支持集对应的文字标签
            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                unique_labels = torch.unique(support_labels)
                support_features = [
                    torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels]
                support_features = torch.stack(support_features)
                context_support = [
                    torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels]
                context_support = torch.stack(context_support)
                # if OPTICAL == True:
                #     support_sub_feature =  [
                #         torch.mean(torch.index_select(support_sub_feature, 0, extract_class_indices(support_labels, c)), dim=0)
                #         for c in unique_labels]
                #     support_sub_feature = torch.stack(support_sub_feature)
            support_features_fusion = torch.cat([support_sub_feature, context_support],dim=1)# stack操作 【5，9，1024】=【5，8，1024】+【5，1，1024】
            context_support_change = context_support.expand(-1, 8, -1)
            support_features = self.context2(support_sub_feature+context_support_change, support_features_fusion, support_features_fusion)[:,:self.args.DATA.NUM_INPUT_FRAMES, :]


            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                pass
            else:
                unique_labels = torch.unique(support_labels)
                support_features = [
                    torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels]
                support_features = torch.stack(support_features)  # [5,8,1024]
            # 对于每个唯一标签c，代码使用extract_class_indices函数从support_labels中提取所有标签为c的样本的索引。然后，使用torch.index_select函数从support_features中选取这些索引对应的特征向量，
            # 并使用torch.mean函数对选取的特征向量进行平均操作，得到一个代表该标签的平均特征向量。支持样本的特征向量通过这种方式被聚合成了一个由唯一标签对应的平均特征向量组成的列表(support_features)

            unique_labels = torch.unique(support_labels)

            n_queries = target_features.shape[0]
            n_support = support_features.shape[0]

            support_features = rearrange(support_features, 'b s d -> (b s) d')  # [40,1024]
            support_sub_feature = rearrange(support_sub_feature, 'b s d -> (b s) d')  # [40,1024]
            target_features = rearrange(target_features, 'b s d -> (b s) d')  # [200,1024]
            target_sub_feature = rearrange(target_sub_feature, 'b s d -> (b s) d')  # [200,1024]

            frame_sim = cos_sim(target_features, support_features)  # [200,40]
            if OPTICAL == True:
                flow_sim = cos_sim(target_sub_feature,support_sub_feature) #[25,5]
            frame_dists = 1 - frame_sim
            flow_dists = 1 - flow_sim
            dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)  # [25, 5, 8, 8]
            flow_dists = rearrange(flow_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)  # [25, 5, 8, 8]

            # calculate query -> support and support -> query

            cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
            cum_flow = OTAM_cum_dist_v2(flow_dists) + OTAM_cum_dist_v2(rearrange(flow_dists, 'tb sb ts ss -> tb sb ss ts'))
        else:
            support_features, target_features, _ = self.get_feats(support_images, target_images, support_labels)
            support_bs = support_features.shape[0]
            target_bs = target_features.shape[0]

            feature_classification_in = torch.cat([support_features, target_features], dim=0)
            feature_classification = self.classification_layer(feature_classification_in).mean(1)
            text_class = self.text_features_test[batch_class.long()]
            class_text_logits = cos_sim(feature_classification, text_class) * self.scale

            if self.training:
                context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)
            else:
                context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)
            if OPTICAL == True:# [5,8,1024]
                with torch.no_grad():
                    support_sub_feature = self.opticalFlow_branch(support_sub_frame.permute(0,2,1,3,4))
                    target_sub_feature = self.opticalFlow_branch(target_sub_frame.permute(0,2,1,3,4))
                support_sub_feature = self.fc_sub(support_sub_feature)
                support_sub_feature = support_sub_feature.unsqueeze(1) * support_features
                support_sub_feature= self.flowTrans(support_sub_feature,support_sub_feature, support_sub_feature)
                target_sub_feature = self.fc_sub(target_sub_feature)
                target_sub_feature = target_sub_feature.unsqueeze(1) * target_features
                target_sub_feature=  self.flowTrans(target_sub_feature,target_sub_feature,target_sub_feature)

            target_features = self.context2(target_features,target_features,target_features)  # 【5，8，1024】 支持集过transformer
            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                unique_labels = torch.unique(support_labels)
                support_features = [
                    torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)),
                               dim=0) for c in unique_labels]
                support_features = torch.stack(support_features)
                context_support = [
                    torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)),
                               dim=0) for c in unique_labels]
                context_support = torch.stack(context_support)
                # if OPTICAL == True:
                    # support_sub_feature = [
                    #     torch.mean(torch.index_select(support_sub_feature, 0, extract_class_indices(support_labels, c)),
                    #                dim=0) for c in unique_labels]
                    # support_sub_feature = torch.stack(support_sub_feature)

            support_features_fusion = torch.cat([support_sub_feature, context_support],dim=1)  # stack操作 【5，9，1024】=【5，8，1024】+【5，1，1024】
            context_support_change = context_support.expand(-1, 8, -1)
            support_features = self.context2(support_sub_feature + context_support_change, support_features_fusion,support_features_fusion)[:, :self.args.DATA.NUM_INPUT_FRAMES, :]

            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                pass
            else:
                unique_labels = torch.unique(support_labels)
                support_features = [
                    torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)),
                               dim=0) for c in unique_labels]
                support_features = torch.stack(support_features)

            unique_labels = torch.unique(support_labels)

            n_queries = target_features.shape[0]
            n_support = support_features.shape[0]

            support_features = rearrange(support_features, 'b s d -> (b s) d')  # [40,1024]
            support_sub_feature = rearrange(support_sub_feature, 'b s d -> (b s) d')  # [40,1024]
            target_features = rearrange(target_features, 'b s d -> (b s) d')  # [200,1024]
            target_sub_feature = rearrange(target_sub_feature, 'b s d -> (b s) d')  # [200,1024]

            frame_sim = cos_sim(target_features, support_features)  # [200,40]
            if OPTICAL == True:
                flow_sim = cos_sim(target_sub_feature, support_sub_feature)  # [25,5]
            frame_dists = 1 - frame_sim
            flow_dists = 1 - flow_sim
            dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries,sb=n_support)  # [25, 5, 8, 8]
            flow_dists = rearrange(flow_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries,sb=n_support)  # [25, 5, 8, 8]

            cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
            cum_flow = OTAM_cum_dist_v2(flow_dists) + OTAM_cum_dist_v2(rearrange(flow_dists, 'tb sb ts ss -> tb sb ss ts'))
        if OPTICAL == True:
            class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]  # list[5]
            flow_sim = [torch.mean(torch.index_select(cum_flow, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
            class_dists = torch.stack(class_dists)  # [5,5]
            flow_sim = torch.stack(flow_sim)
            class_dists = rearrange(class_dists, 'c q -> q c')
            flow_sim = rearrange(flow_sim,'c q -> q c')
            return_dict = {'logits': - class_dists, "class_logits": class_text_logits,"class_flow_logits": - flow_sim }#"temp_logits":temporal_text_logits
        else:
            class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]  # list[5]
            class_dists = torch.stack(class_dists)  # [5,5]
            class_dists = rearrange(class_dists, 'c q -> q c')
            return_dict = {'logits': - class_dists, "class_logits": class_text_logits,"class_flow_logits":False}#"temp_logits":temporal_text_logits


        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())