import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay, classification_report, accuracy_score

import warnings
warnings.filterwarnings("ignore")


def test_statistics(outputs, targets, target_labels=["Negative", "Neutral", "Positive"], target_indices=[-1,0,1], test_loss=None):
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
    cm = confusion_matrix(targets, outputs, labels=target_indices)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=target_labels).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    try:
        cr = classification_report(targets, outputs, target_names=target_labels)
    except:
        cr = classification_report(targets, outputs, target_names=["Negative", "Positive"])

    # Classification Report
    print("Classifcation Report:")
    print(cr)

    return accuracy_score(targets, outputs)