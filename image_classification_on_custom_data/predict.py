import tensorflow as tf
import numpy as np


def predict_new_images(model, class_names, new_img_path, img_height, img_width):
    """
    It loads an image, converts it to an array, expands the dimensions of the array, and then runs the
    image through the model to get a prediction
    :param model: The model that you've trained
    :param class_names: The names of the classes that the model can predict
    :param new_img_path: The path to the image we want to predict
    :param img_height: The height of the image in pixels
    :param img_width: The width of the image in pixels
    """
    new_img_path = "../image_dir/image_name.jpg"

    img = tf.keras.utils.load_img(new_img_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
            class_names[np.argmax(score)], 100 * np.max(score)
        )
    )
    
