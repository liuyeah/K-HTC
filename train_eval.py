import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import logging
import time
import os
import json
import argparse
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import ndcg_score, precision_score, recall_score, f1_score
import pdb

from eval import evaluate


def train(config, model, train_iter, valid_iter, test_iter, id2label, logger, checkpoints_dir):
    result_record = {'valid': {}, 'test': {}}
    best_result_record = {}
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)
    criterion  =  nn.BCELoss()

    best_valid_macro = 0
    best_macro_epoch = 0

    early_stop_count = 0

    for e in range(1, config.num_epoches+1):
        # early stop
        if early_stop_count >= config.early_stop:
            break

        model.train()
        logger.info(" Training epoch: {}".format(e))
        # losses = 0.
        loss_epoch = []
        
        step = 0
        total_step  =  len(train_iter)
        for ind, data in tqdm(enumerate(train_iter), total=total_step):
            texts, text_mask, concepts, neighbors, map_list, overlap_matrix, labels = data

            texts = texts.to(config.device)
            text_mask = text_mask.to(config.device)
            concepts = concepts.to(config.device)

            neighbors = neighbors.to(config.device)
            map_list = map_list.to(config.device)

            overlap_matrix = overlap_matrix.to(config.device)
            labels = labels.to(config.device)

            outputs, conept_CL_loss, label_CL_loss = model(texts, text_mask, concepts, neighbors, map_list, overlap_matrix, labels, status='train')
            model.zero_grad()
            loss  = criterion(outputs, labels.float()) + config.concept_gama * conept_CL_loss + config.label_gama * label_CL_loss

            loss_epoch.append(loss.item())

            loss.backward()
            optimizer.step()
            step += 1
            if step % 1000 == 0:
                logger.info(" \tstep({:>3}/{:>3}) done. Avg Loss:{:.4f}".format(step, total_step, np.mean(loss_epoch)))
                losses  =  0.

        public_micro_f1, public_macro_f1 = test(config, model, valid_iter, id2label)
        logger.info(" \tVALID: public_micro_f1: {:.4f}, public_macro_f1: {:.4f})".format(public_micro_f1, public_macro_f1))
        result_record['valid'][e] = {'micro_f1': public_micro_f1, 'macro_f1': public_macro_f1}

        early_stop_count += 1
        if public_macro_f1 > best_valid_macro:
            best_valid_macro = public_macro_f1
            best_macro_epoch = e
            early_stop_count = 0
            torch.save(model, checkpoints_dir +'/best_macro_model.pt')

        # test result
        public_micro_f1, public_macro_f1 = test(config, model, test_iter, id2label)
        logger.info(" \tTEST: public_micro_f1: {:.4f}, public_macro_f1: {:.4f})".format(public_micro_f1, public_macro_f1))
        result_record['test'][e] = {'micro_f1': public_micro_f1, 'macro_f1': public_macro_f1}

        best_result_record['valid_macro'] = {
            'best_valid_macro_epoch': best_macro_epoch,
            'corresponding test': result_record['test'][best_macro_epoch]
        }
    
    return result_record, best_result_record


def test(config, model, test_iter, id2label):

    model.eval()
    predict_all = torch.Tensor([])
    groundtruth = torch.Tensor([])
    with torch.no_grad():
        for ind, data in enumerate(test_iter):

            texts, text_mask, concepts, neighbors, map_list, overlap_matrix, labels = data

            texts = texts.to(config.device)
            text_mask = text_mask.to(config.device)
            concepts = concepts.to(config.device)

            neighbors = neighbors.to(config.device)
            map_list = map_list.to(config.device)

            outputs = model(texts, text_mask, concepts, neighbors, map_list, concept_overlap_matrix=None, input_labels=None, status='test')

            predict_all = torch.cat([predict_all, outputs.cpu()], dim=0)
            groundtruth = torch.cat([groundtruth, labels.float()], dim=0)
    
    public_micro_f1, public_macro_f1 = public_metrics(predict_all, groundtruth, id2label)

    return public_micro_f1, public_macro_f1


def public_metrics(predict_scores, groundtruth, id2label):
    predict_scores = predict_scores.tolist()
    groundtruth = groundtruth.tolist()
    processed_groundtruth = []
    for line in groundtruth:
        processed_line = []
        for idx in range(len(line)):
            if line[idx] == 1:
                processed_line.append(idx)
        processed_groundtruth.append(processed_line)
    
    processed_id2label = {}
    for key in id2label:
        processed_id2label[int(key)] = id2label[key]
    
    public_results = evaluate(predict_scores, processed_groundtruth, processed_id2label)

    return public_results['micro_f1'], public_results['macro_f1']
    