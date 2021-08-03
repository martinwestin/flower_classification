import tensorflow as tf
import image_processing
import numpy as np
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import sys


CLASSES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
try:
    print("Data loaded from preload file...")
    x, y = pickle.load(open("data.pickle", "rb"))

except IOError:
    print("No preloaded data recognized. Loading data... please wait as this could take a couple minutes...")
    x, y = image_processing.load_data()
    pickle.dump((x, y), open("data.pickle", "wb"))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
y_train = np.array(list(map(lambda x: CLASSES.index(x), y_train)))
y_test = np.array(list(map(lambda x: CLASSES.index(x), y_test)))


def timer(f):
    def decorator(*args, **kwargs):
        start = time.time()
        print(f"Started function \"{f.__name__}\"")
        rv = f(*args, **kwargs)
        end = time.time()
        print(f"Ended function \"{f.__name__}\". Total time: {end - start} seconds")
        return rv

    return decorator


@timer
def augment(x, y):
    augmented_x, augmented_y = [], []
    for i in range(len(x)):
        for j in range(16):
            augmented_image = image_processing.Img.augment(x[i], image_processing.IMAGE_WIDTH,
                                                           image_processing.IMAGE_HEIGHT)
            augmented_x.append(augmented_image[0])
            augmented_y.append(y[i])

    return np.array(augmented_x), np.array(augmented_y)


def new_model(train_x, train_y, val_x, val_y):
    img_size = (image_processing.IMAGE_WIDTH, image_processing.IMAGE_HEIGHT, 3)
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_size,
                                                   include_top=False,
                                                   weights="imagenet")
    # we don't want to change the already set weights and biases that have been reached by the network
    base_model.trainable = False
    # now that we have our base layer setup we can add the classifier. instead of flattening the
    # feature map of the base layer we will use a global average pooling layer that will average the
    # entire 5x5 area of each 2D feature map and return to us a single 1280 element vector per filter.
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dropout_layer = tf.keras.layers.Dropout(0.6)
    dense_layer = tf.keras.layers.Dense(16, activation="relu")
    prediction_layer = tf.keras.layers.Dense(5, activation="softmax")

    model = tf.keras.models.Sequential([
        base_model,
        global_average_layer,
        dropout_layer,
        dense_layer,
        prediction_layer
    ])

    base_learning_rate = 0.0001  # the learning rate --> how much modification of the weights and biases is allowed
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    initial_epochs = 15
    model.fit(train_x, train_y, epochs=initial_epochs, validation_data=(val_x, val_y))
    model.save("model.h5")


if "model.h5" not in os.listdir(sys.path[0]):
    print("Model not found... Retraining...")
    augmented_x, augmented_y = augment(x_train, y_train)
    new_model(augmented_x, augmented_y, x_test, y_test)


def classify_new_image(path):
    image = image_processing.Img(path, (image_processing.IMAGE_WIDTH, image_processing.IMAGE_HEIGHT))
    model = tf.keras.models.load_model("model.h5")
    image.resize()
    pixels = image.get_rbg_values()
    pixels = pixels.reshape(1, image_processing.IMAGE_WIDTH, image_processing.IMAGE_HEIGHT, 3)
    prediction = model.predict(pixels)
    return CLASSES[np.argmax(prediction)]


def ask_retrain_model():
    try:
        retrain = input("Model found. Would you like to retrain anyway (Y/N)? ").lower() == "y"
        if retrain:
            augmented_x, augmented_y = augment(x_train, y_train)
            new_model(augmented_x, augmented_y, x_test, y_test)

    except IOError:
        print("No model found. Retraining...")
        augmented_x, augmented_y = augment(x_train, y_train)
        new_model(augmented_x, augmented_y, x_test, y_test)


if __name__ == '__main__':
    ask_retrain_model()
    model = tf.keras.models.load_model("model.h5")
    predict = model.predict(x_test)

    for i in range(10):
        plt.figure(i)
        plt.imshow(x_test[i])
        plt.title(f"Predicted: {CLASSES[np.argmax(predict[i])]}")
        plt.xlabel(f"Actual: {CLASSES[y_test[i]]}")

    plt.show()
