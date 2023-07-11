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
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay, classification_report, accuracy_score


sys.path.append("../../")


class Logger():
    """
    Klaase die das Logging ermoeglicht.
    Diese speichert alle Informationen im Summarywrite von pytorch
    und gleichzeitig auf der Konsole.
    """

    def __init__(self, name="", locdir="./runs", time_stamp=True, new_model:bool = True):
        self._name = name
        if time_stamp:
            self._name = self._name + \
                (" - " if name != "" else "") + str(datetime.now()).replace(":","-")

        self._model_name = self._name if new_model else name
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

    def save_loss_chart(self, train_loss: list, eval_loss: list, name, step: int):
        """
        Logged ein Bild der atkuellen Klassifiezierung 
        """
        x = range(len(train_loss))
        path = os.path.join(self._locdir, 'loss_charts')

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
        axes[1].set_ylim(bottom=0, top=2)
        for ax in axes:
            ax.plot(x, train_loss, color='b', label='Train Loss')
            ax.plot(x, eval_loss, color='r', label='Evaluation Loss')
            ax.legend()
            ax.set_ylabel('Loss')
            ax.set_xlabel('Epoch')  # Notice the use of set_ to begin methods

        axes[0].set_title('Loss in Training: ' + name)
        axes[1].set_title('Loss in Training (fixed scale): ' + name)
        plt.tight_layout()
        plt.savefig(os.path.join(path, "loss_chart" + "_" + self._name + ".png"))

        # Save image in Logger
        if (step % 250 == 0):
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf)
            image = ToTensor()(image)
            self._logger.add_image("image/class", image, step)

        print("[Logger]: Charts saved.")
        plt.close(fig)

    def save_confussion_chart(self, outputs, targets, step: int):

        fig = plt.figure()
        cm = confusion_matrix(targets, outputs)
        #plt.matshow(cm)
        cm_display = ConfusionMatrixDisplay(cm).plot(cmap=plt.cm.Blues)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image)
        self._logger.add_image("image/confussion_matrix", image, step)

        print("[Logger]: Charts saved.")
        plt.close('all')

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

        torch.save(model, os.path.join(path, filename + "_" + self._model_name + '.pt'))
        #torch.save(state_dict, os.path.join(path, filename + "_" + self._model_name + '.pt'))
        #model_scripted = torch.jit.script(model)  # Export to TorchScript
        #model_scripted.save(os.path.join(path, filename + "_" + self._model_name + '.pt'))

    def close(self):
        """
        Beendet das loggen
        """
        print("[LOG]: Closing logger.")
        self._logger.close()
