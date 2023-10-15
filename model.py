import torch
import config
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv import SAGEConv
import pdb


class New_Masked_Attention(nn.Module):
    def __init__(self, input_units, att_dim, num_classes):
        super(New_Masked_Attention, self).__init__()
        # Attention
        self.linear1 = nn.Linear(input_units, att_dim, bias=False)
        self.linear2 = nn.Linear(att_dim, num_classes, bias=False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.linear2.weight)


    def forward(self, input_x, mask_x):
        # input_x: batch, seq_len, repres_size
        # mask_x: batch, 1, seq_len
        mask_x = torch.unsqueeze(mask_x, dim=1)
        mask_x = 1 - mask_x
        mask_x = mask_x.bool()
        # O_h: batch, seq_len, att_size
        O_h = torch.tanh(self.linear1(input_x))
        # attention_matrix: batch, num_class, seq_len
        attention_matrix = self.linear2(O_h).transpose(1, 2)
        # attention_matrix: batch, num_class, seq_len
        attention_matrix = attention_matrix.masked_fill(mask_x, -float('inf'))
        # attention_weight: batch, num_class, seq_len
        attention_weight = torch.softmax(attention_matrix, dim=2)
        # attention_out: batch, num_class, representation
        attention_out = torch.matmul(attention_weight, input_x)

        # # attention_out: batch, representation
        attention_out = torch.mean(attention_out, dim=1)
        return attention_out


class New_Masked_Label_Attention(nn.Module):
    def __init__(self, input_units, att_dim):
        super(New_Masked_Label_Attention, self).__init__()
        # Attention
        self.linear1 = nn.Linear(input_units, att_dim, bias=False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight, gain=nn.init.calculate_gain('tanh'))

    def forward(self, input_x, mask_x, label_embedding):
        # input_x: batch, seq_len, repres_size
        # label_embedding: num_class, repres_size
        # mask_x: batch, 1, seq_len
        mask_x = torch.unsqueeze(mask_x, dim=1)
        mask_x = 1 - mask_x
        mask_x = mask_x.bool()
        # O_h: batch, seq_len, att_size
        O_h = torch.tanh(self.linear1(input_x))
        # attention_matrix: batch, num_class, seq_len
        attention_matrix = torch.matmul(O_h, label_embedding.transpose(0, 1)).transpose(1, 2)
        # attention_matrix: batch, num_class, seq_len
        attention_matrix = attention_matrix.masked_fill(mask_x, -float('inf'))
        # attention_weight: batch, num_class, seq_len
        attention_weight = torch.softmax(attention_matrix, dim=2)
        # attention_out: batch, num_class, representation
        attention_out = torch.matmul(attention_weight, input_x)

        # # attention_out: batch, representation
        attention_out = torch.mean(attention_out, dim=1)
        return attention_out


class Bert_repres(nn.Module):
    def __init__(self, config, freeze_bert):
        super(Bert_repres, self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_model_path, output_hidden_states=True, return_dict=True)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        

    def forward(self, input_ids, text_mask):

        outputs = self.bert(input_ids = input_ids, attention_mask = text_mask)
        hidden_states = outputs.last_hidden_state
        
        cls_states = outputs.pooler_output
        
        return hidden_states, cls_states


class Label_GCN(torch.nn.Module):
    def __init__(self, repres_size, num_layer=2):
        super(Label_GCN, self).__init__()
        self.num_layer = num_layer
        self.conv1 = GCNConv(repres_size, repres_size)
        if self.num_layer == 2:
            self.conv2 = GCNConv(repres_size, repres_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.num_layer == 2:
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)

        return x


class SAGE(torch.nn.Module):
    def __init__(self, num_channels, num_layer=1, dropout=0.5):
        super(SAGE, self).__init__()
        self.num_layer = num_layer
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(num_channels, num_channels))
        if self.num_layer == 2:
            self.convs.append(SAGEConv(num_channels, num_channels))
        
        self.dropout = dropout
            
    def forward(self, x, edge_index):
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        if self.num_layer == 2:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[1](x, edge_index)
        
        return x


class bert_avg_concept_model(nn.Module):
    def __init__(self, config, concept_embedding, label_texts, label_mask, label_concepts, \
        label_neighbors, label_map_list, pyg_relations):
        super(bert_avg_concept_model, self).__init__()

        self.label_texts = label_texts.to(config.device)
        self.label_mask = label_mask.to(config.device)
        self.label_concepts = label_concepts.to(config.device)
        self.label_neighbors = label_neighbors.to(config.device)
        self.label_map_list = label_map_list.to(config.device)
        self.pyg_relations = pyg_relations.to(config.device)

        self.bert_repres = Bert_repres(config, freeze_bert = False)

        self.att = New_Masked_Attention(input_units=config.bert_size, att_dim=config.label_embedding, num_classes=config.num_classes)
        self.label_att = New_Masked_Label_Attention(input_units=config.bert_size, att_dim=config.bert_size)

        self.concept_embedding = nn.Embedding.from_pretrained(concept_embedding, freeze=False, padding_idx=0)

        self.gcn_module = Label_GCN(config.bert_size, num_layer=config.label_gcn_layer)
        self.sage_module = SAGE(config.bert_size, num_layer=config.concept_sage_layer)

        # contrastive learning
        self.label_tanu = config.label_tanu
        self.concept_tanu = config.concept_tanu

        self.cl_fc = nn.Linear(3*config.bert_size, config.bert_size, bias=True)

        self.classifier = nn.Sequential(
            nn.Linear(config.bert_size, config.bert_size, bias=True),
            nn.ReLU(),
            nn.Linear(config.bert_size, config.num_classes, bias=True),
            nn.Sigmoid()
        )
        # initilization
        def linear_init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        nn.init.xavier_uniform_(self.cl_fc.weight)
        self.classifier.apply(linear_init_weight)


    def obtain_mixed_repres(self, text_ids, text_mask, concept_ids, neighbors, map_list):
        # text_repres: batch, seq_len, repres_size
        text_repres, cls_repres = self.bert_repres(text_ids, text_mask)

        # total_concept_num, repres_size
        total_concept_embedding = self.concept_embedding(map_list)
        # total_concept_num, repres_size
        saged_concept_embedding = self.sage_module(total_concept_embedding, neighbors)

        # concept_repres: batch, seq_len, repres_size
        concept_repres = saged_concept_embedding[concept_ids]

        # final_repres: batch, seq_len, repres_size
        mixed_repres = text_repres + concept_repres

        return mixed_repres, cls_repres
    

    def masked_mean(self, batched_seq_repres, batched_mask):
        # batched_seq_repres: batch, max_len, repres_size
        # batched_mask: batch, max_len

        # batched_mask: batch, max_len, 1
        batched_mask = torch.unsqueeze(batched_mask, dim=-1)
        # masked_repres: batch, max_len, repres_size
        masked_repres = torch.mul(batched_seq_repres, batched_mask)
        # masked_repres: batch, repres_size
        masked_repres = torch.mean(masked_repres, dim=1)

        return masked_repres
    

    def concept_CL_loss(self, input_representation, C, tanu):
        def softmax_wo_exp(x, dim=-1):
            return x / torch.sum(x, dim=dim, keepdim=True)

        Beta = softmax_wo_exp(C, dim=-1)
        Beta = torch.where(torch.isnan(Beta), torch.full_like(Beta, 0), Beta)

        distance = torch.norm(input_representation[:, None] - input_representation, dim=2, p=2)
        distance = torch.exp(- distance / tanu)
        
        diag = torch.diag(distance)
        d_diag = torch.diag_embed(diag)
        distance = distance - d_diag

        score = softmax_wo_exp(distance, dim=-1)
        score = score + torch.eye(score.shape[0]).to(score.device)

        log_score = torch.log(score)

        output_loss = - torch.mul(Beta, log_score)

        output_loss = torch.sum(output_loss)

        return output_loss
    

    def label_CL_loss(self, input_representation, input_labels, tanu):
        input_labels = input_labels.float()
        C = torch.matmul(input_labels, input_labels.transpose(0, 1))
        diag = torch.diag(C)
        c_diag = torch.diag_embed(diag)
        C = C - c_diag

        def softmax_wo_exp(x, dim=-1):
            return x / torch.sum(x, dim=dim, keepdim=True)

        Beta = softmax_wo_exp(C, dim=-1)
        Beta = torch.where(torch.isnan(Beta), torch.full_like(Beta, 0), Beta)

        distance = torch.norm(input_representation[:, None] - input_representation, dim=2, p=2)
        distance = torch.exp(- distance / tanu)
        diag = torch.diag(distance)
        d_diag = torch.diag_embed(diag)
        distance = distance - d_diag

        score = softmax_wo_exp(distance, dim=-1)
        score = score + torch.eye(score.shape[0]).to(score.device)

        log_score = torch.log(score)

        output_loss = - torch.mul(Beta, log_score)

        output_loss = torch.sum(output_loss)

        return output_loss

    
    def forward(self, text_ids, text_mask, concept_ids, neighbors, map_list,\
        concept_overlap_matrix, input_labels, status='train'):

        # document_mixed_repres: batch, seq_len, repres_size
        document_mixed_repres, document_cls_repres = self.obtain_mixed_repres(text_ids, text_mask, concept_ids, neighbors, map_list)
        # label_mixed_repres: num_class, seq_len, repres_size
        label_mixed_repres, _ = self.obtain_mixed_repres(self.label_texts, self.label_mask, self.label_concepts, self.label_neighbors, self.label_map_list)
        # label_mixed_repres: num_class, repres_size
        label_mixed_repres = self.masked_mean(label_mixed_repres, self.label_mask)
        
        root_label_repres = torch.mean(label_mixed_repres, dim=0, keepdim=True)
        # graph_label_repres: num_class+1, repres_size
        graph_label_repres = torch.cat([label_mixed_repres, root_label_repres], dim=0)
        graph_label_repres = self.gcn_module(graph_label_repres, self.pyg_relations)

        # label_propogated_repres: num_class, repres_size
        label_propogated_repres = graph_label_repres[:graph_label_repres.shape[0]-1,:]
        # attention_out: batch, repres_size
        attention_out = self.att(document_mixed_repres, text_mask)
        # label_attention_out: batch, repres_size
        label_attention_out = self.label_att(document_mixed_repres, text_mask, label_propogated_repres)
        # catted_repres: batch, 3*repres_size
        catted_repres = torch.cat([document_cls_repres, label_attention_out, attention_out], dim=-1)
        # to_cl_repres: batch, repres_size
        to_cl_repres = self.cl_fc(catted_repres)
        # output: batch, num_class
        output = self.classifier(to_cl_repres)

        if status == 'train':
            logits_conept_CL_loss = self.concept_CL_loss(to_cl_repres, concept_overlap_matrix, tanu=self.concept_tanu)
            logits_label_CL_loss = self.label_CL_loss(to_cl_repres, input_labels, tanu=self.label_tanu)
            return output, logits_conept_CL_loss, logits_label_CL_loss
        else:
            return output


if __name__ == '__main__':
    config = config.Config()
    model = Bert_repres(config=config, freeze_bert=False)

    input_ids = torch.tensor([[1,2,3,0,0],[2,3,4,6,7]])
    input_mask = torch.tensor([[1,1,1,0,0],[1,1,1,1,1]])

    model(input_ids, input_mask)
