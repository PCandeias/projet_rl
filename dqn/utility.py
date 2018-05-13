from pathlib import Path
from keras.models import clone_model
import numpy as np
import matplotlib.pyplot as plt

models_directory = "models/"
EPS = 1e-8

def file_exists(filename):
    file = Path(filename)
    return file.is_file()

def copy_model(model):
    new_model = clone_model(model)
    new_model.set_weights(model.get_weights())
    return new_model

def plot_results(results):
    plt.plot(np.arange(0, len(results)), results)
    plt.ylabel('score')
    plt.xlabel('episode')
    plt.show()

