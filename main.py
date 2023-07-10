# Imports
import json
import os
import sys

# Torch specific packages
import torch
import torch.nn as nn
# from torchvision import transforms
import torch.optim as optim
from torch import cuda
import transformers
from transformers import BertTokenizer, BertModel, BertConfig
# My intern packages for dataloader, model etc

from models.bert import BERTClass, BERTClass_2FC, BERTClass_res, BERTClass_2FC_5
from models.MLP import Class_MLP_2RES, Class_2FC, Class_FC, Class_2FC_mse
from models.bert_without_mlp import Class_2FC
from nettrainer import NetTrainer

import warnings
warnings.filterwarnings("ignore")

# Function to load json file with all the parameters
# - Model:
# - Dataloader:
# - Trainer:
# - Optimizer:
def load_json(param_file="test_params.json", params_dir="parameters"):
    """
    Laedt eine Json Datei die als erster Parameter uebergeben wurde.
    Gibt das geladene json zurueck.
    """
    #param_file = sys.argv[1] if len(sys.argv) > 1 else param_file
    with open(params_dir + "/" + param_file, "r") as openfile:
        dataholder = json.load(openfile)
    # Return the parameter dictionary
    return dataholder


def main():
    # Laden der Json Parameter
    print("[MAIN]: Loading json file")
    dataholder = load_json("params_bert.json")

    # Device ermitteln (GPU oder CPU)
    use_cuda = dataholder["gpu"]
    device = 'cuda' if cuda.is_available() and use_cuda else 'cpu'

    # Laden modellspezifischer Inhalte
    if dataholder['model_path'] != "":
        file_path = os.path.join("runs", "model_saves", dataholder['model_path'])
        if device == 'cuda':
            model = torch.load(file_path)
        else:
            model = torch.load(file_path, map_location=torch.device('cpu'))
        print("[MAIN]: Model loaded")
    else:
        if dataholder['model_type'] == 'BERT':
            # Laden des Netzes
            model = BERTClass_2FC_5()
        elif (dataholder['model_type'] == 'MLP'):
            tokenizer = None
            model = Class_FC()
        print("[MAIN]: Model created")

    if dataholder['model_type'] == 'BERT':
        # Tokenizer für BERT laden
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if dataholder["freeze_first"]:
        for param in model.l1.parameters():
            param.requires_grad = False
    else:
        for param in model.l1.parameters():
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters (total): {total_params}")
    print(f"Number of parameters (trainale): {trainable_params}")

    model.to(device)

    # Laden der Loss Function
    print("[MAIN]: Loading criterion")
    criterion = None
    if dataholder["criterion"] == "mse":
        criterion = nn.MSELoss()
    elif dataholder["criterion"] == "crossentropy":
        criterion = nn.CrossEntropyLoss()

    # Error Check
    if model is None:
        raise Exception("[Dataholder]: No proper 'net' found!")
    if criterion is None:
        raise Exception("[Dataholder]: No proper 'criterion' found!")

    # Set training parameters
    dataset_params = {
        'onehot': dataholder["onehot"],
        'tokenize_bert': dataholder["tokenize"],
        'max_len': dataholder["max_len"],
        'tokenizer': tokenizer,
        'onehot_encoding': [0,1,2,3,4]
    }
    # Trainer erzeugen
    print("[MAIN]: Loading trainer")
    trainer = NetTrainer(model,
                         new_model=(dataholder['model_path'] != ""),
                         batchsize_train=dataholder["batchsize_train"],
                         batchsize_val=dataholder["batchsize_val"],
                         model_type=dataholder['model_type'],
                         criterion=criterion,
                         seed=dataholder.get("seed"),
                         device=device,
                         name=dataholder.get("name"),
                         dataholder_str=json.dumps(dataholder, indent=4),
                         dataset_params=dataset_params,
                         path_sets=os.path.join("data", dataholder["data_dir"]))



    # Laden Optimizer
    print("[MAIN]: Loading optimizer")
    optimizer = None
    if dataholder["optimizer"] == "sge":
        optimizer = optim.SGD(model.parameters(),
                              lr=dataholder["learning_rate"],
                              momentum=dataholder["momentum"])
    elif dataholder["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters())

    # Check nach Fehlern beim Optimizer laden
    if optimizer is None:
        raise Exception("[Dataholder]: No proper 'optimizer' found!")
    # Start Training
    print("[MAIN]: Start Training")
    trainer.train(dataholder.get("epochs"),
                  optimizer,
                  dataholder.get("n_trainsets"),
                  dataholder.get("start_trainset"),
                  dataholder.get("patience"),
                  dataholder.get("inflation"))


if __name__ == '__main__':
    main()
