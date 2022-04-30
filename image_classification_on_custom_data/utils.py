import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools

import matplotlib.font_manager as fm

fprop = fm.FontProperties(fname="../jp_font/NotoSansJP-Regular.otf")


def visualize_training(history, epochs):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    # plt.show()
    # save the plot
    plt.savefig(f"../output/{epochs}/train_valid_graph.png")
    # close the plot
    plt.close()


def fn_plot_confusion_matrix(
    cm, classes, epochs, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)
    plt.figure(figsize=(20, 20))

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontproperties=fprop)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontproperties=fprop)
    plt.yticks(tick_marks, classes, fontproperties=fprop)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label", fontproperties=fprop)
    plt.xlabel("Predicted label", fontproperties=fprop)

    # save the plot
    plt.savefig(f"../output/{epochs}/confusion_matrix.png")
    # close the plot
    plt.close()
