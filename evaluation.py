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


def test(model, path_testset, dataset_params, device, criterion):
    """
    Diese Funktion testet ein Netz nach dem Training mit dem Testdatensatz
    :param model: Modell, dass zu testen ist
    :param path_testset: Pfad zum Testset
    :param dataset_params: Dataset parameter als dict
    :param device: 'GPU' oder 'CPU'
    :param criterion: Loss function
    :return: -
    """

    # Rohdaten als dataframe laden
    test_data = pd.read_csv(path_testset, delimiter=";")
    test_data = test_data.iloc[:100].reset_index(drop=True)

    # Initialisierung Dataset und Dataloader
    test_dataset = dataset(test_data["Phrase"], test_data["Sentiment"], **dataset_params)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True,
                             num_workers=0)
    # Gradienten nicht berechnen bei Validation
    with torch.no_grad():
        for batch in test_loader: #Batch umfasst gesamten Datensatz
            print('[Test]: Prediction...')
            # TODO: Input statt batch übergeben
            prediction_scores = model(batch, device)
            # TODO: über if parameter, ob onthot oder nicht?
            prediction_labels = torch.argmax(prediction_scores, dim=1)
            targets_labels = torch.argmax(batch['targets'], dim=1)

            # Calculate Loss
            total_loss = criterion(prediction_scores, batch['targets'])
            avg_loss = total_loss / len(test_dataset)
            print(avg_loss)

            # confusion matrix
            cm = confusion_matrix(targets_labels, prediction_labels)
            cm_display = ConfusionMatrixDisplay(cm).plot()
            plt.show()

            # Classification Report
            print(classification_report(targets_labels, prediction_labels))




def main_bert():
    # Laden der Json Parameter
    print("[MAIN]: Loading json file")
    dataholder = load_json("params_bert.json")

    path_test = "./data/datasets/Testset.csv"

    use_cuda = dataholder['gpu']
    device = 'cuda' if cuda.is_available() and use_cuda else 'cpu'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BERTClass()
    model.load_state_dict(torch.load(dataholder['model_path']))
    model.to(device)

    dataset_params = {
        'onehot': dataholder["onehot"],
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

    test(model, path_test, dataset_params, device, criterion)


if __name__ == '__main__':
    main_bert()
