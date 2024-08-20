#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""Config class for loading and keeping track of the configs."""

import os
os.chdir("/home/imi1214/data1/xiaqing/Vit_adapter_clip_FSAR")
import sys
sys.path.append('/home/imi1214/data1/xiaqing/Vit_adapter_clip_FSAR')
import yaml
import json
import copy
import argparse
import utils.checkpoint as ckp

import utils.logging as logging
logger = logging.get_logger(__name__)

class Config(object):
    """
    Global config object. 
    It automatically loads from a hierarchy of config files and turns the keys to the 
    class attributes. 
    """
    def __init__(self, load=True, cfg_dict=None, cfg_level=None):
        """
        Args: 
            load (bool): whether or not yaml is needed to be loaded.
            cfg_dict (dict): dictionary of configs to be updated into the attributes
            cfg_level (int): indicating the depth level of the config
        """
        self._level = "cfg" + ("." + cfg_level if cfg_level is not None else "")
        if load:
            self.args = self._parse_args()
            print("Loading config from {}.".format(self.args.cfg_file)) # 加载参数文件
            self.need_initialization = True
            cfg_base = self._initialize_cfg() # 读取./configs/pool/base.yaml配置文件
            cfg_dict = self._load_yaml(self.args) # 读取选定的配置文件，在config里
            cfg_dict = self._merge_cfg_from_base(cfg_base, cfg_dict) # 将配置都荷载一起
            self.cfg_dict = cfg_dict
        self._update_dict(cfg_dict) #   将字典的内容递归地设置为配置类对象的属性，并且把load改成false了
        if load:
            ckp.make_checkpoint_dir(self.OUTPUT_DIR)

    def _parse_args(self):
        """
        Wrapper for argument parser. 
        """
        parser = argparse.ArgumentParser(
            description="Argparser for configuring [code base name to think of] codebase"
        )
        parser.add_argument(
            "--cfg",
            dest="cfg_file",
            help="Path to the configuration file",
            default="configs/projects/CLIPFSAR/kinetics100/CLIPFSAR_K100_1shot_v1.yaml"
        )
        parser.add_argument(
            "--init_method",
            help="Initialization method, includes TCP or shared file-system",
            default="tcp://localhost:9999",
            type=str,
        )
        parser.add_argument(
            "opts",
            help="other configurations",
            default=None,
            nargs=argparse.REMAINDER
        )
        return parser.parse_args()

    def _path_join(self, path_list):
        """
        Join a list of paths.
        Args:
            path_list (list): list of paths.
        """
        path = ""
        for p in path_list:
            path+= p + '/'
        return path[:-1]

    def _initialize_cfg(self):
        """
        When loading config for the first time, base config is required to be read.
        """
        if self.need_initialization:
            self.need_initialization = False
            if os.path.exists('./configs/pool/base.yaml'):
                with open("./configs/pool/base.yaml", 'r') as f:
                    cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
            else:
                # for compatibility to the cluster
                with open("./DAMO-Action/configs/pool/base.yaml", 'r') as f:
                    cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return cfg
    
    def _load_yaml(self, args, file_name=""):
        """
        Load the specified yaml file.
        Args:
            args: parsed args by `self._parse_args`.
            file_name (str): the file name to be read from if specified.
        """
        assert args.cfg_file is not None
        if not file_name == "": # reading from base file
            with open(file_name, 'r') as f:
                cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        else: # reading from top file
            with open(args.cfg_file, 'r') as f:
                cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
                file_name = args.cfg_file

        if "_BASE_RUN" not in cfg.keys() and "_BASE_MODEL" not in cfg.keys() and "_BASE" not in cfg.keys():
            # return cfg if the base file is being accessed
            return cfg

        if "_BASE" in cfg.keys():
            # load the base file of the current config file
            if cfg["_BASE"][1] == '.':
                prev_count = cfg["_BASE"].count('..') # 根据配置文件，嵌套生成路径继续读取
                cfg_base_file = self._path_join(file_name.split('/')[:(-1-cfg["_BASE"].count('..'))] + cfg["_BASE"].split('/')[prev_count:])
            else:
                cfg_base_file = cfg["_BASE"].replace(
                    "./", 
                    args.cfg_file.replace(args.cfg_file.split('/')[-1], "")
                )
            cfg_base = self._load_yaml(args, cfg_base_file)
            cfg = self._merge_cfg_from_base(cfg_base, cfg)
        else:
            # load the base run and the base model file of the current config file
            if "_BASE_RUN" in cfg.keys():
                if cfg["_BASE_RUN"][1] == '.':
                    prev_count = cfg["_BASE_RUN"].count('..')
                    cfg_base_file = self._path_join(file_name.split('/')[:(-1-prev_count)] + cfg["_BASE_RUN"].split('/')[prev_count:])
                else:
                    cfg_base_file = cfg["_BASE_RUN"].replace(
                        "./", 
                        args.cfg_file.replace(args.cfg_file.split('/')[-1], "")
                    )
                cfg_base = self._load_yaml(args, cfg_base_file)
                cfg = self._merge_cfg_from_base(cfg_base, cfg, preserve_base=True) # 合起来，保护base
            if "_BASE_MODEL" in cfg.keys():
                if cfg["_BASE_MODEL"][1] == '.':
                    prev_count = cfg["_BASE_MODEL"].count('..')
                    cfg_base_file = self._path_join(file_name.split('/')[:(-1-cfg["_BASE_MODEL"].count('..'))] + cfg["_BASE_MODEL"].split('/')[prev_count:])
                else:
                    cfg_base_file = cfg["_BASE_MODEL"].replace(
                        "./", 
                        args.cfg_file.replace(args.cfg_file.split('/')[-1], "")
                    )
                cfg_base = self._load_yaml(args, cfg_base_file)
                cfg = self._merge_cfg_from_base(cfg_base, cfg)
        cfg = self._merge_cfg_from_command(args, cfg)
        return cfg
    
    def _merge_cfg_from_base(self, cfg_base, cfg_new, preserve_base=False):
        """
        Replace the attributes in the base config by the values in the coming config, 用即将到来的配置中的值替换基本配置中的属性
        unless preserve base is set to True.除非preserve-base设置为True。
        Args:
            cfg_base (dict): the base config.基本配置。
            cfg_new (dict): the coming config to be merged with the base config.即将与基本配置合并的配置。
            preserve_base (bool): if true, the keys and the values in the cfg_new will 如果为true，则cfg_new中的键和值将不替换cfgbase中的键和值，如果它们存在于cfg_base。
                not replace the keys and the values in the cfg_base, if they exist in 当密钥和值不存在于cfg_base中时，然后将它们填充到cfgbase中。
                cfg_base. When the keys and the values are not present in the cfg_base,
                then they are filled into the cfg_base.
                用即将到来的配置中的值替换基本配置中的属性，

        """
        for k,v in cfg_new.items(): # 以new为主，把new的值填充到base去
            if k in cfg_base.keys():
                if isinstance(v, dict):
                    self._merge_cfg_from_base(cfg_base[k], v)
                else:
                    cfg_base[k] = v
            else:
                if "BASE" not in k or preserve_base:
                    cfg_base[k] = v
        return cfg_base

    def _merge_cfg_from_command(self, args, cfg):
        """
        Merge cfg from command. Currently only support depth of four. 用即将到来的配置中的值替换基本配置中的属性，
        E.g. VIDEO.BACKBONE.BRANCH.XXXX. is an attribute with depth of four.
        Args:
            args : the command in which the overriding attributes are set.
            cfg (dict): the loaded cfg from files.
        """
        assert len(args.opts) % 2 == 0, 'Override list {} has odd length: {}.'.format(
            args.opts, len(args.opts)
        )
        keys = args.opts[0::2]
        vals = args.opts[1::2]

        # maximum supported depth 3
        for idx, key in enumerate(keys):
            key_split = key.split('.')
            assert len(key_split) <= 4, 'Key depth error. \nMaximum depth: 3\n Get depth: {}'.format(
                len(key_split)
            )
            assert key_split[0] in cfg.keys(), 'Non-existant key: {}.'.format(
                key_split[0]
            )
            if len(key_split) == 2:
                assert key_split[1] in cfg[key_split[0]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
            elif len(key_split) == 3:
                assert key_split[1] in cfg[key_split[0]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
                assert key_split[2] in cfg[key_split[0]][key_split[1]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
            elif len(key_split) == 4:
                assert key_split[1] in cfg[key_split[0]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
                assert key_split[2] in cfg[key_split[0]][key_split[1]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
                assert key_split[3] in cfg[key_split[0]][key_split[1]][key_split[2]].keys(), 'Non-existant key: {}.'.format(
                    key
                )


            if len(key_split) == 1:
                cfg[key_split[0]] = vals[idx]
            elif len(key_split) == 2:
                cfg[key_split[0]][key_split[1]] = vals[idx]
            elif len(key_split) == 3:
                cfg[key_split[0]][key_split[1]][key_split[2]] = vals[idx]
            elif len(key_split) == 4:
                cfg[key_split[0]][key_split[1]][key_split[2]][key_split[3]] = vals[idx]
            
        return cfg
    
    def _update_dict(self, cfg_dict):
        """
        Set the dict to be attributes of the config recurrently.
        Args:
            cfg_dict (dict): the dictionary to be set as the attribute of the current 
                config class.
        """
        def recur(key, elem):
            if type(elem) is dict:
                return key, Config(load=False, cfg_dict=elem, cfg_level=key)
            else:
                if type(elem) is str and elem[1:3]=="e-":
                    elem = float(elem)
                return key, elem
        
        dic = dict(recur(k, v) for k, v in cfg_dict.items())
        self.__dict__.update(dic)
    
    def get_args(self):
        """
        Returns the read arguments.
        """
        return self.args
    
    def __repr__(self):
        return "{}\n".format(self.dump())
            
    def dump(self):
        return json.dumps(self.cfg_dict, indent=2) # 将对象是属性转换成json格式，缩进为2 打印信息

    def deep_copy(self):
        return copy.deepcopy(self)
    
if __name__ == '__main__':
    # debug
    cfg = Config(load=True)
    print(cfg.DATA)
    