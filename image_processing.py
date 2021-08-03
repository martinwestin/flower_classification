import os
from PIL import Image
import numpy as np
import tensorflow as tf


IMAGE_HEIGHT, IMAGE_WIDTH = (128, 128)
DIRECTORIES = [
        os.path.join("flowers", "daisy"), os.path.join("flowers", "dandelion"),
        os.path.join("flowers", "rose"), os.path.join("flowers", "sunflower"),
        os.path.join("flowers", "tulip")
    ]


class Img:
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    def __init__(self, path, dimensions: tuple):
        self.path = path
        self.width, self.height = dimensions

    def get_rbg_values(self):
        img = Image.open(self.path)
        img = img.convert("RGB")  # make sure that there is no alpha channel
        pixel_values = np.array(list(img.getdata()))
        pixel_values = pixel_values.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3) / 255
        return pixel_values

    def resize(self):
        img = Image.open(self.path)
        img = img.resize((self.width, self.height))
        img.save(self.path)

    @classmethod
    def augment(cls, pixels: np.ndarray, img_width: int, img_height: int):
        img = pixels.reshape(1, img_width, img_height, 3)
        return cls.data_augmentation(img)


def get_image_paths():
    files = []
    for directory in DIRECTORIES:
        for file in os.listdir(directory):
            files.append(os.path.join(directory, file))

    return files


def resize_images():
    for path in get_image_paths():
        img = Img(path, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img.resize()


def load_data():
    """
    load the image data with corresponding labels.
    :return: 2 numpy arrays (features and labels).
    """
    x = []
    y = []
    for path in get_image_paths():
        if len(list(filter(lambda i: i == path.split("/")[-2], y))) < 730:
            img = Img(path, (IMAGE_WIDTH, IMAGE_HEIGHT))
            x.append(img.get_rbg_values())
            y.append(path.split("/")[-2])

    return np.array(x), np.array(y)
