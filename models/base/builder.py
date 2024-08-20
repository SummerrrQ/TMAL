#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Builder for video models. """

import sys
import torch
import torch.nn as nn

import traceback

import utils.logging as logging

from models.base.models import BaseVideoModel, MODEL_REGISTRY # 注册表，把basemodel也导入
from models.utils.model_ema import ModelEmaV2

logger = logging.get_logger(__name__)

def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (Config): global config object that provides specifics to construct the model.
        gpu_id (Optional[int]): specify the gpu index to build model.
    Returns:
        model: constructed model
        model_ema: copied model for ema
    """
    # Construct the model  models/base/models.py 注册表中找对应config里的模型名字
    if MODEL_REGISTRY.get(cfg.MODEL.NAME) == None: # 检查给定的模型名称是否在 @MODEL_REGISTRY.register()后面定义，config前面加get的方法
        # attempt to find standard models
        model = BaseVideoModel(cfg)
    else:
        # if the model is explicitly defined,
        # it is directly constructed from the model pool
        model = MODEL_REGISTRY.get(cfg.MODEL.NAME)(cfg)

    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = 0 #gpu_id
        model = model.cuda(device=cur_device)
    
    model_ema = None
    if cfg.MODEL.EMA.ENABLE: # 创建一个指数移动平均模型
        model_ema = ModelEmaV2(model, decay=cfg.MODEL.EMA.DECAY)

    try:
        # convert batchnorm to be synchronized across 
        # different GPUs if needed
        sync_bn = cfg.BN.SYNC_BN # 将批归一化（Batch Normalization，BN）层转换为在多个 GPU 上进行同步，以实现跨不同 GPU 的批归一化的功能。
        if sync_bn == True and cfg.NUM_GPUS * cfg.NUM_SHARDS > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    except:
        sync_bn = None

    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS*cfg.NUM_SHARDS > 1:
        # Make model replica operate on the current device
        if cfg.PAI:
            # Support distributed training on the cluster
            model = torch.nn.parallel.DistributedDataParallel(
                module=model
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device
            )

    return model, model_ema