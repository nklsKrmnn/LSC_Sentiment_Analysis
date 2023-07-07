import os

import pandas as pd

from models.bert_without_mlp import BERTClass_without_mlp
from torch import cuda
import numpy as np
from data.datasets import dataset as dataset
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import transformers
from transformers import BertTokenizer, BertModel, BertConfig

def preprocess_first_layer(path_sets, file_path):
  with torch.no_grad():
    device = 'cuda' if cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERTClass_without_mlp()
    model.to(device)

    path = os.path.join(path_sets, file_path)
    data = pd.read_csv(path, delimiter=";")
    data = data.reset_index(drop=True)

    dataset_params = {
            'onehot': True,
            'onehot_encoding': [-1, 0, 1],
            'tokenize_bert': True,
            'max_len': 200,
            'tokenizer': tokenizer
        }

    dataset_tmp = dataset(data["Phrase"], data["Sentiment"], **dataset_params)


    # Dataloader initialisieren mit Datasets
    loader = DataLoader(dataset_tmp, batch_size=1)

    outputs_bert = []
    for _, batchdata in enumerate(loader):
      outputs_bert.append(model(batchdata, device).tolist()[0])

    outputs_bert = np.array(outputs_bert)
    dataset_after_first_layer = dataset(outputs_bert,
            data["Sentiment"],
            tokenize_bert=False,
            onehot=True,
            second_layer=True)
    dataset_after_first_layer[3]

    return dataset_after_first_layer

path_sets = "./dataset_mr"
train_file = "Trainset_complete.csv"

dataset_after_first_layer = preprocess_first_layer(path_sets, train_file)
new_path = "data/dataset_mr_after_first_layer_notOH/" + train_file.split('.')[0] + ".pt"
torch.save(dataset_after_first_layer, new_path)
print(train_file.split('.')[0] + " saved")