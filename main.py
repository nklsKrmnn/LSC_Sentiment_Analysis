# Imports
import json
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

from models.bert import BERTClass, BERTClass_2FC
from nettrainer import NetTrainer


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
    dataholder = load_json("params_bert_2FC.json")

    # Device ermitteln (GPU oder CPU)
    use_cuda = dataholder["gpu"]
    device = 'cuda' if cuda.is_available() and use_cuda else 'cpu'



    # Laden modellspezifischer Inhalte
    if dataholder['model_type'] == 'BERT':
        # Tokenizer für BERT laden
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Laden des Netzes
        model = BERTClass_2FC()
        if dataholder['model_path'] != "":
            model.load_state_dict(torch.load(dataholder['model_path']))
        model.to(device)

    model.l1.requires_grad_(False)

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
        'tokenizer': tokenizer
    }

    # Trainer erzeugen
    print("[MAIN]: Loading trainer")
    trainer = NetTrainer(model,
                         batchsize_train=dataholder["batchsize_train"],
                         batchsize_val=dataholder["batchsize_val"],
                         model_type=dataholder['model_type'],
                         criterion=criterion,
                         seed=dataholder.get("seed"),
                         device=device,
                         name=dataholder.get("name"),
                         dataholder_str=json.dumps(dataholder, indent=4),
                         dataset_params=dataset_params,
                         path_sets="./data/datasets_mr")



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