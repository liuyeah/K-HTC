#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/11/07 20:18:11
@Author  :   liuyeah 
@Version :   1.0
'''

# here put the import lib
import re
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import json
import time
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from config import Config
from dataset import MyDataset, collate, merge_neighbors
from matplotlib.font_manager import json_dump, json_load
from model import bert_avg_concept_model
from train_eval import train
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
from graph_contruction import construct_bidirected_graph, construct_directed_graph


def seed_everything(seed=0):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.backends.cudnn.deterministic = True


def get_parser(config):
    parser = argparse.ArgumentParser(description='Logging Demo')
    parser.add_argument('--not-save', default=False, action='store_true',
                          help='If yes, only output log to terminal.')

    parser.add_argument('--work-dir', default='./checkpoints/' + config.dataset_name + '/' + config.model_name + '/',
                        help='the work folder for storing results')
    return parser


def loadLogger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)

    logger.addHandler(sHandler)

    if not args.not_save:
        work_dir = os.path.join(args.work_dir,
                                time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()))
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
        fHandler.setLevel(logging.DEBUG)
        fHandler.setFormatter(formatter)

        logger.addHandler(fHandler)

    return logger, work_dir


def load_data(input_filepath):
    texts = []
    entities = []
    neighbors = []
    labels = []
    with open(input_filepath, 'r') as f_in:
        for line in f_in:
            processed_line = json.loads(line)
            texts.append(processed_line['token'])
            entities.append(processed_line['concept'])
            neighbors.append(processed_line['concept_neighbor'])
            labels.append(processed_line['labels'])
    
    return texts, entities, neighbors, labels


def load_wos_name(input_filepath):
    with open(input_filepath, 'r') as f_in:
        input_data = json.load(f_in)
    
    label_name = input_data
    texts = label_name['token']
    entities = label_name['concept']
    neighbors = label_name['concept_neighbor']

    return texts, entities, neighbors


def label_text_concept_pad(label_texts, label_concepts, label_neighbors):
    tensor_label_texts = []

    # label text
    for line in label_texts:
        tensor_label_texts.append(torch.tensor(line))
    
    processed_label_neighbors = []
    for line in label_neighbors:
        processed_label_neighbors.extend(line)
    
    label_concepts, processed_label_neighbors, map_list = merge_neighbors(processed_label_neighbors, label_concepts)

    text_len = torch.tensor([v.size(0) for v in tensor_label_texts])
    text_mask = torch.arange(torch.max(text_len))[None, :] < text_len[:, None]
    text_mask = text_mask.int()

    return pad_sequence(tensor_label_texts, batch_first=True, padding_value=0.0), text_mask, \
        pad_sequence(label_concepts, batch_first=True, padding_value=0.0), torch.tensor(processed_label_neighbors), torch.tensor(map_list)


if __name__ == '__main__':

    config = Config()

    seed_everything(seed=config.seed)

    parser = get_parser(config)
    args = parser.parse_args()
    logger, checkpoints_dir = loadLogger(args)

    train_texts, train_entities, train_neighbors, train_labels = load_data(config.dataset+'wos_train.json')
    valid_texts, valid_entities, valid_neighbors, valid_labels = load_data(config.dataset+'wos_valid.json')
    test_texts, test_entities, test_neighbors, test_labels = load_data(config.dataset+'wos_test.json')

    label_texts, label_entities, label_neighbors = load_wos_name(config.dataset+'wos_label_name.json')

    tensor_label_texts, tensor_label_mask, tensor_label_entities, tensor_label_neighbors, tensor_label_map_list \
        = label_text_concept_pad(label_texts, label_entities, label_neighbors)

    # id2label dic
    with open(config.dataset+'id2label.json', 'r') as f_in:
        id2label = json.load(f_in)
    
    with open(config.dataset+'entity_embedding.json') as f_in:
        entity_embedding = json.load(f_in)
    
    pyg_relations = torch.tensor(construct_bidirected_graph(config.dataset+'wos_label_relation.json', config=config))

    train_dataset = MyDataset(train_texts, train_entities, train_neighbors, train_labels)
    train_data = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)

    valid_dataset = MyDataset(valid_texts, valid_entities, valid_neighbors, valid_labels)
    valid_data = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)

    test_dataset = MyDataset(test_texts, test_entities, test_neighbors, test_labels)
    test_data = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)

    logger.info("load data successfully")

    entity_embedding = torch.tensor(entity_embedding)
    our_model = bert_avg_concept_model(config, entity_embedding, tensor_label_texts, tensor_label_mask, tensor_label_entities, \
        tensor_label_neighbors, tensor_label_map_list, pyg_relations)
    our_model.to(config.device)

    logger.info("start training!")

    printed_config = vars(config)
    printed_config['device'] = str(printed_config['device'])
    logger.info(json.dumps(printed_config, indent=True))

    result_record, best_result_record = train(config, our_model, train_data, valid_data, test_data, id2label, logger, checkpoints_dir)

    # logger.info(result_record)

    logger.info('*'*20 + ' result ' + '*'*20)
    logger.info(json.dumps(best_result_record, indent=True))

