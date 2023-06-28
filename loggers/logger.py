from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from PIL import Image
import numpy as np
import os
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import io
import torch
import sys

sys.path.append("../../")


class Logger():
    """
    Klaase die das Logging ermoeglicht.
    Diese speichert alle Informationen im Summarywrite von pytorch
    und gleichzeitig auf der Konsole.
    """

    def __init__(self, name="", locdir="./runs", time_stamp=True):
        self._name = name
        if time_stamp:
            self._name = self._name + \
                (" - " if name != "" else "") + str(datetime.now()).replace(":","-")

        self._locdir = locdir
        self._logger = SummaryWriter(locdir + "/" + self._name)

    def log_string(self, desc, text):
        """
        Logged ein einen String TEXT im Token DESC
        """
        self._logger.add_text(desc, text)

    def train_start(self):
        """
        Logged den Startzeitpunkt des Trainings
        """
        print("[LOG]: Training startet.")
        self._trainstart = datetime.now()
        self._logger.add_text("trainduration/start", str(self._trainstart))

    def train_end(self, reason):
        """
        Logged den Endzeitpunkt und Dauer des Trainings
        """
        trainend = datetime.now()
        self._logger.add_text("trainduration/end", str(trainend))
        self._logger.add_text("trainduration/duration",
                              str(trainend - self._trainstart))
        self._logger.add_text("trainduration/reason", reason)
        print("[LOG]: Training finished! Runtime {}, because of {}".format(
            (trainend - self._trainstart), reason))

    def val_loss(self, value, step):
        """
        Logged den Loss der Validation
        """
        print("[LOG]: Validation Step {} logged. Loss {}".format(step, value))
        self._logger.add_scalar("loss/val", value, step)

    def train_loss(self, value, step):
        """
        Logged den Loss der Trainings
        """
        print("[LOG]: Training Step {} logged. Loss {}".format(step, value))
        self._logger.add_scalar("loss/train", value, step)

    def val_acc(self, value, step):
        """
        Logged den Loss der Trainings
        """
        print("[LOG]: Training Step {} logged. Accuracy {}".format(step, value))
        self._logger.add_scalar("Accuracy/val", value, step)

    def model_log(self, model, input_data=None):
        """
        Logged das Model als Interactiven Graphen
        """
        model.eval()
        with torch.no_grad():
            self._logger.add_graph(model, input_data, False, False)

    def model_text(self, model):
        """
        Logged das Model in textueller Form
        """
        self._logger.add_text("model", str(model))

    def summary(self, category, desc):
        """
        Logged eine Zusammenfassung des Trainings
        """
        self._logger.add_text("summary" + "/" + category, str(desc))

    def save_loss_chart(self, train_loss: list, eval_loss: list, name):
        """
        Logged ein Bild der atkuellen Klassifiezierung 
        """
        x = range(len(train_loss))
        path = os.path.join(self._locdir, 'loss_charts')

        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.plot(x, train_loss, color='b', label='Train Loss')
        axes.plot(x, eval_loss, color='r', label='Evaluation Loss')
        axes.legend()
        axes.set_xlabel('Epoch')  # Notice the use of set_ to begin methods
        axes.set_ylabel('Loss')
        axes.set_title('Loss in Training:' + name)
        plt.savefig(os.path.join(path, "loss_chart" + "_" + self._name + ".png"))
        print("[Logger]: Chart saved.")
        plt.close(fig)
        
    def save_net(self, model, filename="best_model"):
        """
        Speichert das Trainierte Netz
        """
        path = os.path.join(self._locdir, 'model_saves')
        if (not os.path.exists(path)):
            os.makedirs(path)
        try:
            state_dict = model.state_dict()
        except AttributeError:
            state_dict = model.module.state_dict()
        torch.save(state_dict, os.path.join(path, filename + "_" + self._name + "_"+'.pt'))

    def close(self):
        """
        Beendet das loggen
        """
        print("[LOG]: Closing logger.")
        self._logger.close()
