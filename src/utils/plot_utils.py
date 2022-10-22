import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    experiment_dir,
    classes,
    preds,
    trues,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    save=False,
    filename="CM",
    ext="pdf",
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """

    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(10, 10))
    plt.rcParams["font.size"] = 20
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tick_params(labelsize=20)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    # print(cm)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    if save:
        os.makedirs(f"{experiment_dir}/", exist_ok=True)
        plt.savefig(f"{experiment_dir}/{filename}.{ext}", format=ext, dpi=1000)
