import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch import cuda
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay, classification_report

from models.bert import BERTClass
from data.datasets import dataset
from main import load_json
from evaluation import test_statistics


import warnings
warnings.filterwarnings("ignore")


def test_bert(dataholder, path_testset):
    """
    Diese Funktion testet ein Netz nach dem Training mit dem Testdatensatz
    :param model: Modell, dass zu testen ist
    :param path_testset: Pfad zum Testset
    :param dataset_params: Dataset parameter als dict
    :param device: 'GPU' oder 'CPU'
    :param criterion: Loss function
    :return: -
    """
    use_cuda = dataholder['gpu']
    device = 'cuda' if cuda.is_available() and use_cuda else 'cpu'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    file_path = os.path.join("runs", "model_saves", dataholder['model_path'])

    model = torch.load(file_path, map_location=device)
    model.to(device)

    # Set training parameters
    dataset_params = {
        'onehot': dataholder["onehot"],
        'onehot_encoding': [-1,0,1],
        'tokenize_bert': dataholder["tokenize"],
        'max_len': dataholder["max_len"],
        'tokenizer': tokenizer
    }

    # Laden der Loss Function
    print("[MAIN]: Loading criterion")
    criterion = None
    if dataholder["criterion"] == "mse":
        criterion = nn.MSELoss()
    elif dataholder["criterion"] == "crossentropy":
        criterion = nn.CrossEntropyLoss()

    # Rohdaten als dataframe laden
    test_data = pd.read_csv(path_testset, delimiter=";")
    test_data = test_data.iloc[:50].reset_index(drop=True)

    # Initialisierung Dataset und Dataloader
    test_dataset = dataset(test_data["Phrase"], test_data["Sentiment"], **dataset_params)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True,
                             num_workers=0)

    model.eval()

    fin_targets = []
    fin_outputs = []
    total_loss = 0
    test_step = 0

    # Gradienten nicht berechnen bei Validation
    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(data, device)
            total_loss += criterion(outputs, targets).item()
            test_step += 1
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    outputs, targets = fin_outputs, fin_targets

    # Labels extrahieren
    if (dataset_params['onehot']):
        outputs = np.array(outputs).argmax(axis=1)
        targets = np.array(targets).argmax(axis=1)

    # Calculate Loss
    avg_loss = total_loss / test_step

    return outputs, targets, avg_loss

def main_bert():
    # Laden der Json Parameter
    print("[MAIN]: Loading json file")
    dataholder = load_json("params_bert_test.json")

    path_test = "data/dataset_tw/Testset.csv"

    target_labels = ["negative", "neutral", "positive"]

    outputs, targets, test_loss = test_bert(dataholder, path_test)
    test_statistics(outputs, targets, test_loss=test_loss)

if __name__ == '__main__':
    main_bert()
