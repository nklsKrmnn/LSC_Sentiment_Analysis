import numpy as np
import torch
import torch.utils.data as torchData
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from loggers.logger import Logger
import os
import pandas as pd
from sklearn import metrics

from data.datasets import dataset as dataset

import warnings
warnings.filterwarnings("ignore")

CSV_FILES = False

class NetTrainer():
    """
        Diese Klasse fuehrt das Training durch.
        """

    def __init__(self, model, model_type: str, batchsize_train, batchsize_val, path_sets, dataset_params, criterion,
                 seed=69, device='cpu',
                 name="", dataholder_str=""):
        # Parameter initialisierung
        self.model = model
        self.model_type = model_type
        self.batch_size_train = batchsize_train
        self.batch_size_val = batchsize_val
        self.criterion = criterion
        self.seed = seed
        self.device = device
        self.name = name
        self.path_sets = path_sets
        self.dataset_params = dataset_params

        # Listen zur Sammlung von Train und Validation loss pro Epoch
        self.train_loss = []
        self.validation_loss = []

        # RNG Seed setzen
        torch.manual_seed(self.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(self.seed)

        # Erzeugen des Loggers
        self.logger = Logger(name)

        # Erste Informationen loggen
        self.logger.summary("dataholder", dataholder_str)
        self.logger.model_text(self.model)
        self.logger.summary("seed", self.seed)

        # Initialisierung der Grafikkarte
        if self.device == 'cuda':
            self.gpu_setup()

    def prepare_data(self):
        """
        Lädt das Trainset und das Validationset für das Training. Welche Datensätze es lädt hängt von
        der Initialisierung des Parameters 'path_sets' ab.
        :return: train_loader, val_loader
        """

        if CSV_FILES:
            # Paths ermitteln
            if (self.current_set == -1):
                train_file = "Trainset_complete.csv"
            else:
                train_file = "Trainset_" + str(self.current_set + 1) + ".csv"
            train_path = os.path.join(self.path_sets, train_file)
            val_path = os.path.join(self.path_sets, "Validationset.csv")

            # Rohdaten als dataframe laden
            train_data = pd.read_csv(train_path, delimiter=";")
            val_data = pd.read_csv(val_path, delimiter=";")
            train_data = train_data.reset_index(drop=True)
            val_data = val_data.reset_index(drop=True)

            # Datasets initialisieren mit Rohdaten
            train_dataset = dataset(train_data["Phrase"], train_data["Sentiment"], **self.dataset_params)
            val_dataset = dataset(val_data["Phrase"], val_data["Sentiment"], **self.dataset_params)

        else:
            if (self.current_set == -1):
                train_file = "Trainset_complete.pt"
            else:
                train_file = "Trainset_" + str(self.current_set + 1) + ".pt"
            train_path = os.path.join(self.path_sets, train_file)
            val_path = os.path.join(self.path_sets, "Validationset.pt")
            train_dataset = torch.load(train_path)
            val_dataset = torch.load(val_path)

        # Dataloader initialisieren mit Datasets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size_train, shuffle=True,
                                  num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size_val, shuffle=True,
                                num_workers=0)

        return train_loader, val_loader

    def calc_batch(self, batch_data):
        """
        Diese Funktion berechnet den Loss für eine Batch
        :param batch_data: Daten eine Trainingsbatch. Ein Item ist ein dict{input, targets}
        :return: Loss der Batch.
        """

        outputs = self.model(batch_data, self.device)
        self.optimizer.zero_grad()
        loss = self.criterion(outputs, batch_data['targets'].to(self.device, dtype=torch.float))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def calc_epoch(self, epoch, train_loader):
        """
        Diese Funktion berechnet den durchschnittlichen Loss pro Epoche
        :param epoch: Nummer der Epoche für print-Ausgabe
        :param train_loader: Dataloader mit Trainingsdaten
        :return: Gesamt-Loss geteilt durch Anzahl der Batches.
        """
        # Modell in Trainingsmodus versetzen
        self.model.train()

        epoch_loss = 0
        step_count = 0

        # Iterieren über alle Batches im Dataloader
        for _, batch_data in enumerate(train_loader, 0):

            # Berechnung loss
            epoch_loss += self.calc_batch(batch_data)
            step_count += 1

            # Zwischenausgabe und Sicherheitsspeicherung
            if _ % 20 == 0:
                self.logger.save_net(self.model)
                print(f"Epoch {epoch} Batch {step_count} Loss: {epoch_loss / step_count}")
                print("Model saved")

        return epoch_loss / step_count

    def train(self, epochs, optimizer, n_trainsets, start_set: int = 1, patience=0, inflation=1):
        """
        Methode fuer das ganze Training
        """
        # Log den Start des Trainings
        self.logger.train_start()


        self.current_set = start_set
        self.optimizer = optimizer
        self.inflation = inflation

        train_loss, eval_loss = [], []

        # Daten fuer Early stopping
        # TODO: raus oder nutzen
        min_loss = float('inf')
        cur_patience = 0
        finish_reason = 'Training did not start'

        # Bei -1 gesamtes Trainingsdatenset laden
        if (self.current_set == -1):
            train_loader, test_loader = self.prepare_data()

        # Iterieren über alle Epochen
        for epoch in range(epochs):
            try:
                if (self.current_set > 0):
                    # Pro Epoche Daten neu laden
                    train_loader, test_loader = self.prepare_data()
                    # pro Epoche das nächste Dataset selektieren
                    self.current_set = (self.current_set + 1) % n_trainsets

                # Berechnung Loss und Optimization
                epoch_train_loss = self.calc_epoch(epoch, train_loader)
                self.logger.train_loss(epoch_train_loss, epoch)
                self.train_loss.append(epoch_train_loss)
                train_loss.append(epoch_train_loss)

                # Validation loss berechnen
                epoch_validation_loss, epoch_val_acc = self.validation(test_loader)
                # Log validateion_loss
                self.logger.val_loss(epoch_validation_loss, epoch)
                self.logger.val_acc(epoch_val_acc, epoch)
                self.validation_loss.append(epoch_validation_loss)
                eval_loss.append(epoch_validation_loss)
                self.logger.save_net(self.model)

                self.logger.save_loss_chart(train_loss, eval_loss, self.name, epoch)

            except KeyboardInterrupt:
                # Fuer den Fall wir wollen das Training haendisch abbrechen
                finish_reason = 'Training interrupted by user'
                break
            else:
                finish_reason = 'Training finished normally'

        # Log finish
        self.logger.train_end(finish_reason)

        # Training Cleanup
        self.logger.close()

    def validation(self, test_loader):

        # Modell in eval Modus versetzen
        self.model.eval()

        fin_targets = []
        fin_outputs = []
        val_loss = 0
        val_step = 0

        #Gradienten nicht berechnen bei Validation
        with torch.no_grad():
            for _, data in enumerate(test_loader, 0):
                targets = data['targets'].to(self.device, dtype=torch.float)
                outputs = self.model(data, self.device)
                val_loss += self.criterion(outputs, targets).item()
                val_step += 1
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        outputs, targets = fin_outputs, fin_targets

        # Labels extrahieren
        outputs = np.array(outputs).argmax(axis=1)
        targets = np.array(targets).argmax(axis=1)

        # Metrics berechnen
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")

        return (val_loss / val_step, accuracy)

    def gpu_setup(self):
        """
        FUnktion die das Model und das Criterion auf die GPU laedt.
        """

        print("[TRAINER]: Write model and criterion on GPU")
        self.model.to('cuda')
        self.criterion.to('cuda')

        # Log if cpu or gpu is used
        self.logger.log_string("device usage", ("GPU" if self.device=="GPU" else "CPU"))