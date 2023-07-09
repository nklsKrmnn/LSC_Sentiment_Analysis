import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay, classification_report, accuracy_score

import warnings
warnings.filterwarnings("ignore")


def test_statistics(outputs, targets, target_labels=["Negative", "Neutral", "Positive"], test_loss=None):
    """
    Eine Funktion zur standardisierten Bewertung von Klassifikationsergebnissen.
    Prints: Eine Heatmap einer Confussionmatrix sowie einen Classification-Report.
    :param outputs: Predictions der Testdaten aus dem Modell.
    :param targets: Korrekte Klassifikationen der Testdaten.
    :param target_labels: Labels der Klassifikationen
    :param test_loss (optional): Loss der mit ausgegeben werden soll.
    :return: Accuracy des Prediction
    """

    if test_loss != None:
        print(f'Test-Loss: {test_loss}')

    # confusion matrix
    cm = confusion_matrix(targets, outputs, labels=[-1,0,1])
    cm_display = ConfusionMatrixDisplay(cm, display_labels=target_labels).plot()
    plt.title("Confusion Matrix")
    plt.show()

    cr = classification_report(targets, outputs, target_names=target_labels)

    # Classification Report
    print("Classifcation Report:")
    print(cr)

    return accuracy_score(targets, outputs)


# Teststatistik für Tweet Datensatz mit nur zwei Klassen
def test_statistics_tw(outputs, targets, target_labels=["Negative", "Positive"], test_loss=None):
    """
    Eine Funktion zur standardisierten Bewertung von Klassifikationsergebnissen.
    Prints: Eine Heatmap einer Confussionmatrix sowie einen Classification-Report.
    :param outputs: Predictions der Testdaten aus dem Modell.
    :param targets: Korrekte Klassifikationen der Testdaten.
    :param target_labels: Labels der Klassifikationen
    :param test_loss (optional): Loss der mit ausgegeben werden soll.
    :return: Accuracy des Prediction
    """

    if test_loss != None:
        print(f'Test-Loss: {test_loss}')

    # confusion matrix
    cm = confusion_matrix(targets, outputs, labels=[-1,1])
    cm_display = ConfusionMatrixDisplay(cm, display_labels=target_labels).plot()
    plt.title("Confusion Matrix")
    plt.show()

    cr = classification_report(targets, outputs, target_names=target_labels)

    # Classification Report
    print("Classifcation Report:")
    print(cr)

    return accuracy_score(targets, outputs)