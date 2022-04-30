import pathlib
import os

# hide tf warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import PIL
import tensorflow as tf
import itertools

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def load_data(data_dir):
    """
    It takes a directory, counts the number of images in it, and returns the directory
    :param data_dir: The path to the directory where the images are stored
    :return: The data_dir is being returned.
    """
    data_dir = pathlib.Path(data_dir)
    try:
        # count the number of images in the directory
        image_count = len(list(data_dir.glob("*/*.jpg")))
        print("Found {} images.".format(image_count))
    except:
        raise Exception(
            "No images found in {} or images are not in jpg format".format(data_dir)
        )
    return data_dir


def create_dataset(data_dir, batch_size, h, w, validation_split):
    """
    `create_dataset` takes in a data directory, batch size, image height, image width, and validation
    split, and returns a training dataset and a validation dataset
    :param data_dir: The path to the directory containing the images
    :param batch_size: The number of images to read in at a time
    :param h: height of the image
    :param w: width of the image
    :param validation_split: The percentage of images to be used as a validation set
    :return: A tuple of two datasets, one for training and one for validation.
    """
    batch_size = batch_size
    img_height = h
    img_width = w

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        label_mode="categorical",
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        label_mode="categorical",
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    return train_ds, val_ds


def fn_data_augmentation(img_height, img_width):
    """
    It creates a data augmentation pipeline that randomly flips, rotates, zooms, and adjusts the
    contrast of the input images
    :param img_height: The height of the image
    :param img_width: The width of the image in pixels
    :return: A sequential model with 4 layers.
    """
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip(
                "horizontal", input_shape=(img_height, img_width, 3)
            ),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
            layers.experimental.preprocessing.RandomContrast(0.1),
        ]
    )
    return data_augmentation
