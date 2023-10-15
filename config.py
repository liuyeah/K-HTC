import torch

class Config(object):
    def __init__(self):
        self.seed = 0
        self.sever = 'sever-name'

        self.dataset = './processed_data'
        self.dataset_name = 'wos'
        self.model_name = "k-htc"
        self.num_classes = 141
        self.num_classes_level0 = 7
        self.num_classes_level1 = 134

        self.batch_size = 16
        self.bert_size = 768
        self.learning_rate = 2e-5
        self.num_epoches = 100

        self.label_embedding = 768

        self.concept_sage_layer = 1
        
        self.label_gcn_layer = 1

        self.predict_threshold = 0.5

        self.early_stop = 10

        self.concept_tanu = 10

        self.concept_gama = 0.01

        self.label_tanu = 1

        self.label_gama = 0.0001

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        
        self.bert_model_path = './hf_models/bert-base-uncased/'

        
