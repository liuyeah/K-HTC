## K-HTC

Source code for our paper: "Enhancing Hierarchical Text Classification through Knowledge Graph Integration" [1]


## Data Preparation
### HTC dataset
1. BlurbGenreCollection-EN (BGC) Dataset: https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html
2. Web-of-Science (WOS)[2] Dataset: https://data.mendeley.com/datasets/9rw3vkcfy4/2

### Knowledge Graph
1. We adopt the advanced knowledge graph named ConceptNet [3].

### Data Process
1. Following the strategy proposed by KagNet[4] to recognize the concepts in the documents. Please refer to the original codes: https://github.com/INK-USC/KagNet.
2. After recognizing the concepts, adopt the OpenKE[5] to train concept embedding via TransE.


## Model
After we obtain the processed data, we could run the main.py (we give the example on WOS dataset).

## Citation

>[1] Ye Liu, Kai Zhang, et al. Enhancing Hierarchical Text Classification through Knowledge Graph Integration[C]//Findings of the Association for Computational Linguistics: ACL 2023. 2023: 5797-5810.\
>[2] Kamran Kowsari, Donald E Brown, Mojtaba Heidarysafa, Kiana Jafari Meimandi, Matthew S Gerber, and Laura E Barnes. 2017. Hdltex: Hierarchical deep learning for text classification. In 2017 16th IEEE international conference on machine learning and applications (ICMLA), pages 364–371. IEEE.\
>[3] Robyn Speer, Joshua Chin, and Catherine Havasi. 2017. Conceptnet 5.5: An open multilingual graph of general knowledge. In Thirty-first AAAI conference on artificial intelligence.\
>[4] Bill Yuchen Lin, Xinyue Chen, Jamin Chen, and Xiang Ren. 2019. Kagnet: Knowledge-aware graph networks for commonsense reasoning. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2829–2839.\
>[5] Xu Han, Shulin Cao, Xin Lv, Yankai Lin, Zhiyuan Liu, Maosong Sun, and Juanzi Li. 2018. Openke: An open toolkit for knowledge embedding. In Proceedings of the 2018 conference on empirical methods in natural language processing: system demonstrations, pages 139–144.