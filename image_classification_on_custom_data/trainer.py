from data_generator import create_dataset, fn_data_augmentation, load_data
from base_model import create_model
from utils import visualize_training, fn_plot_confusion_matrix
from predict import predict_new_images

import os
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        required=True,
        help="Path of dataset containing jpg format images",
    )

    parser.add_argument(
        "--epochs", "-e", type=int, default=25, help="Number of epochs to train"
    )

    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=32,
        help="batch size to train",
    )

    parser.add_argument(
        "--img_height",
        "-ih",
        type=int,
        default=224,
        help="height of the image",
    )

    parser.add_argument(
        "--img_width",
        "-iw",
        type=int,
        default=224,
        help="width of the image",
    )

    parser.add_argument(
        "--validation_split",
        "-vs",
        type=float,
        default=0.2,
        help="validation split",
    )

    parser.add_argument(
        "--optimizer",
        "-opt",
        type=str,
        default="adam",
        help="Optimizer to use",
    )

    args = parser.parse_args()
    data_dir = args.data_path
    data_dir = load_data(data_dir)

    ### Hyperparameters ###################
    epochs = args.epochs
    batch_size = args.batch_size
    img_height = args.img_height
    img_width = args.img_width
    validation_split = args.validation_split
    optimizer = args.optimizer
    ### Hyperparameters ends   ##############

    os.makedirs(f"output/{epochs}", exist_ok=True)

    train_ds, val_ds = create_dataset(
        data_dir, batch_size, img_height, img_width, validation_split
    )

    class_names = train_ds.class_names
    print("[INFO] Class names:")
    print(class_names)

    # buffered prefetching so we can yield data from disk without having I/O become blocking.
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)
    print("[INFO] Number of classes:")
    print(num_classes)

    data_augmentation = fn_data_augmentation(img_height, img_width)
    model = create_model(num_classes, data_augmentation)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # train the model
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    visualize_training(history, epochs)

    # # predict new images
    # new_img_path = "test_images/image_name.jpg"
    # predict_new_images(model, class_names, new_img_path, img_height, img_width)

    # evaluate the model on validation data
    loss, accuracy = model.evaluate(val_ds)
    print("[INFO] Validation Loss: {}".format(loss))
    print("[INFO] Validation Accuracy: {}".format(accuracy))

    # print the confusion matrix
    predictions = model.predict(val_ds)

    y_true = []
    y_pred = []
    for x, y in val_ds:
        y = tf.argmax(y, axis=1)
        y_true.append(y)
        y_pred.append(tf.argmax(model.predict(x), axis=1))

    y_pred = tf.concat(y_pred, axis=0)
    y_true = tf.concat(y_true, axis=0)

    print(classification_report(y_true, y_pred, target_names=class_names))

    # print f1 score
    print(f1_score(y_true, y_pred, average="macro"))

    # save the classification report in a file
    with open(f"output/{epochs}/classification_report_{epochs}.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))
        # append f1 score
        f.write(f"\n\n\nF1 score (macro): {f1_score(y_true, y_pred, average='macro')}")
        # append weighted f1 score
        f.write(
            f"\nF1 score (weighted): {f1_score(y_true, y_pred, average='weighted')}"
        )

    # plot the confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)

    fn_plot_confusion_matrix(cnf_matrix, class_names, epochs)


if __name__ == "__main__":
    main()
