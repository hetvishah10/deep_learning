import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_learning_rate_vs_loss(history):
    # Get the actual number of epochs from the history
    epochs = range(1, len(history.history['loss']) + 1)
    lrs = 1e-5 * (10 ** (np.arange(len(epochs))/20))
    
    plt.figure(figsize=(10, 7))
    plt.semilogx(lrs, history.history["loss"])
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning rate vs. loss")
    plt.show()

def plot_training_curves(history):
    pd.DataFrame(history.history).plot()
    plt.title("Model Training Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.show()