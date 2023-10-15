#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   graph_contruction.py
@Time    :   2022/11/19 15:04:43
@Author  :   liuyeah 
@Version :   1.0
'''

# here put the import lib
import re
import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from matplotlib.font_manager import json_dump, json_load

from config import Config


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def construct_bidirected_graph(input_relation_filepath, config):
    with open(input_relation_filepath, 'r') as f_in:
        origin_relations = json.load(f_in)
    
    add_root = config.num_classes
    for node in range(config.num_classes_level0):
        origin_relations.append([add_root, node])

    pyg_relation_h = []
    pyg_relation_t = []
    for relation in origin_relations:
        # 正向
        pyg_relation_h.append(relation[0])
        pyg_relation_t.append(relation[1])
        # 反向
        pyg_relation_h.append(relation[1])
        pyg_relation_t.append(relation[0])
    
    pyg_relation = [pyg_relation_h, pyg_relation_t]

    return pyg_relation


def construct_directed_graph(input_relation_filepath, config):
    with open(input_relation_filepath, 'r') as f_in:
        origin_relations = json.load(f_in)
    
    add_root = config.num_classes
    for node in range(config.num_classes_level0):
        origin_relations.append([add_root, node])

    pyg_relation_h = []
    pyg_relation_t = []
    for relation in origin_relations:
        # 正向
        pyg_relation_h.append(relation[0])
        pyg_relation_t.append(relation[1])
    
    pyg_relation = [pyg_relation_h, pyg_relation_t]

    return pyg_relation


if __name__ == '__main__':
    seed_everything()

    config = Config()

    relation_filepath = config.dataset+'wos_label_relation.json'

    construct_directed_graph(relation_filepath, config)
