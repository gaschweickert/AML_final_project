from typing import Iterator, List, Union, Tuple
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import History
from tensorflow.keras import optimizers
import imageio
import numpy as np
import cv2
import os

import collect_data

import tensorflow as tf
print(tf.version.VERSION)

# Configuring settings for multi-gpu strategy
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    number_of_gpus = len(gpus)
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        mirrored_strategy = tf.distribute.MirroredStrategy()
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)

def split_data(df: pd.DataFrame, split_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Accepts a Pandas DataFrame and splits it into training, testing and validation data. Returns DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Your Pandas DataFrame containing all your data.

    Returns
    -------
    Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        [description]
    """
    train, val = train_test_split(df, test_size=split_size, random_state=1)  # split the data with a validation size o 20%

    print("shape train: ", train.shape)  # type: ignore
    print("shape val: ", val.shape)  # type: ignore

    return train, val  # type: ignore

def get_callbacks(model_name: str) -> List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]:
    """Accepts the model name as a string and returns multiple callbacks for training the keras model.

    Parameters
    ----------
    model_name : str
        The name of the model as a string.

    Returns
    -------
    List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]
        A list of multiple keras callbacks.
    """
    logdir = (
        "logs/scalars/" + model_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )  # create a folder for each model.
    tensorboard_callback = TensorBoard(log_dir=logdir)
    # use tensorboard --logdir logs/scalars in your command line to startup tensorboard with the correct logs

    early_stopping_callback = EarlyStopping(
        monitor="val_mean_absolute_percentage_error",
        min_delta=1,  # model should improve by at least 1%
        patience=10,  # amount of epochs  with improvements worse than 1% until the model stops
        verbose=2,
        mode="min",
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )

    return [tensorboard_callback, early_stopping_callback]

def create_generators(train: pd.DataFrame, val: pd.DataFrame) -> Tuple[Iterator, Iterator, Iterator]:
    """Accepts four Pandas DataFrames: all your data, the training, validation and test DataFrames. Creates and returns
    keras ImageDataGenerators. Within this function you can also visualize the augmentations of the ImageDataGenerators.

    Parameters
    ----------
    df : pd.DataFrame
        Your Pandas DataFrame containing all your data.
    train : pd.DataFrame
        Your Pandas DataFrame containing your training data.
    val : pd.DataFrame
        Your Pandas DataFrame containing your validation data.
    test : pd.DataFrame
        Your Pandas DataFrame containing your testing data.

    Returns
    -------
    Tuple[Iterator, Iterator, Iterator]
        keras ImageDataGenerators used for training, validating and testing of your models.
    """
    train_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        #rotation_range=5,
        #width_shift_range=0.1,
        #height_shift_range=0.1,
        #brightness_range=(0.75, 1),
        #shear_range=0.1,
        #zoom_range=[0.75, 1],
        #horizontal_flip=True,
        #vertical_flip=True,
        #validation_split=0.2,
    )  # create an ImageDataGenerator with multiple image augmentations
    validation_generator = ImageDataGenerator(
        rescale= 1.0 / 255
    )  # except for rescaling, no augmentations are needed for validation and testing generators
    # test_generator = ImageDataGenerator(rescale=1.0 / 255)
    # visualize image augmentations
    # if visualize_augmentations == True:
    #     visualize_augmentations(train_generator, df)

    train_generator = train_generator.flow_from_dataframe(
        dataframe=train,
        x_col="image_loc",  # this is where your image data is stored
        y_col="count",  # this is your target features
        class_mode="raw",  # use "raw" for regressions
        target_size=(224, 224),
        batch_size=128, # increase or decrease to fit your GPU
    )

    validation_generator = validation_generator.flow_from_dataframe(
        dataframe=val,
        x_col="image_loc",
        y_col="count",
        class_mode="raw",
        target_size=(224, 224),
        batch_size=128,
    )
    # test_generator = test_generator.flow_from_dataframe(
    #     dataframe=test,
    #     x_col="image_location",
    #     y_col="count",
    #     class_mode="raw",
    #     target_size=(224, 224),
    #     batch_size=128,
    # )
    return train_generator, validation_generator

def small_cnn() -> Sequential:
    """A very small custom convolutional neural network with image input dimensions of 224x224x3.

    Returns
    -------
    Sequential
        The keras Sequential model.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))

    return model


def run_model(
    model_name: str,
    model_function: Model,
    lr: float,
    train_generator: Iterator,
    validation_generator: Iterator,
) -> History:
    """This function runs a keras model with the Ranger optimizer and multiple callbacks. The model is evaluated within
    training through the validation generator and afterwards one final time on the test generator.

    Parameters
    ----------
    model_name : str
        The name of the model as a string.
    model_function : Model
        Keras model function like small_cnn()  or adapt_efficient_net().
    lr : float
        Learning rate.
    train_generator : Iterator
        keras ImageDataGenerators for the training data.
    validation_generator : Iterator
        keras ImageDataGenerators for the validation data.

    Returns
    -------
    History
        The history of the keras model as a History object. To access it as a Dict, use history.history. For an example
        see plot_results().
    """

    callbacks = get_callbacks(model_name)
    model = model_function
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001), loss="mean_absolute_error", metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()]
    )
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        callbacks=callbacks,
        workers=6, # adjust this according to the number of CPU cores of your machine
    )

    # model.evaluate(
    #     test_generator,
    #     callbacks=callbacks,
    # )

    return history  # type: ignore


def visualize_image(df, idx=1):
    im = cv2.imread(df["image_loc"].iloc[idx])
    cv2.imshow(df["image_loc"].iloc[1], im)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows()



def run(small_sample=False):
    """Run all the code of this file.

    Parameters
    ----------
    small_sample : bool, optional
        If you just want to check if the code is working, set small_sample to True, by default False
    """
    # directory = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    grouped_train_df, train_df, data_abw, data_pbw = (
        pd.read_csv('./dataframes/grouped_data_df.csv'), 
        pd.read_csv('./dataframes/train_df.csv'), 
        pd.read_csv('./dataframes/data_abw.csv'), 
        pd.read_csv('./dataframes/data_pbw.csv') 
    )  

    if small_sample:
        data_abw = data_abw.iloc[0:1000]  # set small_sampe to True if you want to check if your code works without long waiting
    
    train, val = split_data(data_abw, 0.2)  # split your data

    visualize_image(train, 4)

    train_generator, validation_generator = create_generators(train=train, val=val)
    
    small_cnn_history = run_model(
        model_name="small_cnn",
        model_function=small_cnn(),
        lr=0.001,
        train_generator=train_generator,
        validation_generator=validation_generator,
    )

    


if __name__ == "__main__":
    run(small_sample=False)



