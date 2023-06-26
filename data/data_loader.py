import os

import pandas as pd
import torch
import torch.utils.data as data
import numpy as np
from torch.utils.data import Dataset, DataLoader

# eigene Klassen
from data.datasets import dataset

class data_loader_BERT():
    """
    Eine Klasse, die die verschiedenen Datensätze lädt und vorbereitet zu Verarbeitung im Training.

    Diese Dataloader Klasse muss je nach Modell hinsichtlich der Preprocessing pipeline angepasst werden.
    """

    def __init__(self, batch_size_train, batch_size_val, path_sets: str, shuffle: bool = True, start_dataset=1, tokenizer=None, max_len_token=None):
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.path_sets = path_sets
        self.shuffle = shuffle
        self.current_set = start_dataset - 1
        self.tokenizer = tokenizer
        self.max_len_token = max_len_token


    def next_set(self):
        self.current_set = (self.current_set + 1) % 6

    def prepare_dataset(self, dataset_typ: str = "BERT", onehot: bool = True):
        """
        Dataset splitten und in Validation und Trainingset und in Dataloader stecken.
        Über den Parameter "dataset_typ" kann die preprocessing pipeline definiert werden.
        """
        train_file = "Trainset_" + str(self.current_set + 1) + ".csv"
        train_path = os.path.join(self.path_sets, train_file)
        val_path = os.path.join(self.path_sets, "Validationset.csv")

        train_data = pd.read_csv(train_path, delimiter=";")
        val_data = pd.read_csv(val_path, delimiter=";")

        train_data = train_data.reset_index(drop=True)
        val_data = val_data.reset_index(drop=True)

        prepper = prepper_bert(onehot)

        train_dataset = prepper.prep_data(train_data, self.tokenizer, self.max_len_token)
        val_dataset = prepper.prep_data(val_data, self.tokenizer, self.max_len_token)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size_train, shuffle=self.shuffle,
              num_workers= 0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size_val, shuffle=self.shuffle,
              num_workers= 0)

        return train_loader, val_loader

    def prepare_testset(self, dataset_typ: str = "BERT", onehot: bool = True):
        """
        Dataset splitten und in Validation und Trainingset und in Dataloader stecken.
        Über den Parameter "dataset_typ" kann die preprocessing pipeline definiert werden.
        """
        test_path = os.path.join(self.path_sets, "Testset.csv")

        test_data = pd.read_csv(test_path, delimiter=";")

        test_data = test_data.reset_index(drop=True)

        prepper = prepper_bert(onehot)

        test_dataset = prepper.prep_data(test_data)

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size_val, shuffle=self.shuffle,
                                num_workers=0)

        return test_loader