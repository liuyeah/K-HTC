import torch
import pickle
import numpy as np
from config import Config
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset


def merge_neighbors(neighbors, concepts):
    relations = []
    for relation in neighbors:
        if relation not in relations:
            relations.append(relation)
    
    generated_dic = {0: 0}
    for line_concept in concepts:
        line_concept = set(line_concept)
        for concept in line_concept:
            if concept not in generated_dic:
                generated_dic[concept] = len(generated_dic)
    
    for relation in relations:
        if relation[0] not in generated_dic:
            generated_dic[relation[0]] = len(generated_dic)
        if relation[1] not in generated_dic:
            generated_dic[relation[1]] = len(generated_dic)
    
    output_concepts = []
    for line_concept in concepts:
        mapped_concept = list(map(lambda item: generated_dic[item], line_concept))
        output_concepts.append(torch.tensor(mapped_concept))

    output_relations = list(map(lambda item: [generated_dic[item[0]], generated_dic[item[1]]], relations))
    pyg_relations = [[], []]
    for relation in output_relations:
        pyg_relations[0].append(relation[0])
        pyg_relations[1].append(relation[1])

    map_list = [0] * len(generated_dic)
    for old_id in generated_dic:
        new_id = generated_dic[old_id]
        map_list[new_id] = old_id
    
    return output_concepts, pyg_relations, map_list


def overlap_num(concept_set_list):
    # batch, batch
    overlap_matrix = [[0 for col in range(len(concept_set_list))] for row in range(len(concept_set_list))]
    for idx in range(len(concept_set_list)):
        for jdx in range(len(concept_set_list)):
            if idx == jdx:
                continue
            overlap_matrix[idx][jdx] = len(concept_set_list[idx].intersection(concept_set_list[jdx]))
    
    # pdb.set_trace()
    return overlap_matrix


# only utilize text infomation
def collate(examples):
    texts, concepts, concept_sets, neighbors, labels = [], [], [], [], []
    for item in examples:
        texts.append(item[0])
        concepts.append(item[1])

        temp_concept_set = set(item[1])
        temp_concept_set.discard(0)
        concept_sets.append(temp_concept_set)

        neighbors.extend(item[2])
        labels.append(item[3])
    
    concepts, neighbors, map_list = merge_neighbors(neighbors, concepts)

    text_len = torch.tensor([v.size(0) for v in texts])
    text_mask = torch.arange(torch.max(text_len))[None, :] < text_len[:, None]
    text_mask = text_mask.int()

    overlap_matrix = overlap_num(concept_sets)

    # padding
    return pad_sequence(texts, batch_first=True, padding_value=0.0), text_mask, \
        pad_sequence(concepts, batch_first=True, padding_value=0.0), torch.tensor(neighbors), torch.tensor(map_list), \
        torch.tensor(overlap_matrix), torch.stack(labels)


class MyDataset(Dataset):
    def __init__(self, text, concept, neighbor, label):
        super(MyDataset, self).__init__()
        self.text = text
        self.concept = concept
        self.neighbor = neighbor
        self.label = label
        self.len = len(label)

    def __getitem__(self, index: int):
        return torch.tensor(self.text[index]), self.concept[index], self.neighbor[index], torch.tensor(self.label[index])
    
    def __len__(self):
        return self.len

