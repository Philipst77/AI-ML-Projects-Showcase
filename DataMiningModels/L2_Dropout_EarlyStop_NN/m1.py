import numpy as np
import tensorflow as tf
import barPlotTemplate as brp
import confusionMatrixHeatmap as cnf
import ConvertToOddEven as ote
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras import utils as np_utils
from keras.models import load_model


from pa2pre1 import processTestData
import argparse


def parseArguments():
    parser = argparse.ArgumentParser(
        description='Build a Keras model for Image classification')

    parser.add_argument('--training_x', action='store',
                        dest='XFile', default="", required=True,
                        help='matrix of training images in npy')
    parser.add_argument('--training_y', action='store',
                        dest='yFile', default="", required=True,
                        help='labels for training set')

    parser.add_argument('--outModelFile', action='store',
                        dest='outModelFile', default="", required=True,
                        help='model name for your Keras model')

    return parser.parse_args()

def main():
    np.random.seed(1671)

    parms = parseArguments()

    X_train = np.load(parms.XFile)
    y_train = np.load(parms.yFile)

    (X_train, y_train) = processTestData(X_train,y_train)

    print('KERA modeling build starting...')
    ## Build your model here
    model1 = Sequential()
    model1.add(Dense(500, activation="relu", input_shape=(784,)))
    model1.add(Dense(500, activation="relu"))
    model1.add(Dense(10, activation="softmax"))
    model1.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model1.fit(X_train, y_train, epochs  = 250, verbose=1 )
    model1.save(parms.outModelFile)
    X_val = np.load("MNIST_X_test_1.npy")
    y_val = np.load("MNIST_y_test_1.npy")
    (X_val, y_val) = processTestData(X_val, y_val)
    y_pred_probs = model1.predict(X_val) 
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    cnf.drawConfusionMatrix(conf_matrix, list(range(10)))
    countholder = np.bincount(y_pred_classes, minlength=10)
    brp.drawBarPlot(countholder, list(range(10)))
    y_odd_even = ote.convertToOddEven(y_pred_classes)
    counts_oe = np.bincount(y_odd_even, minlength=2)
    brp.drawBarPlot(counts_oe, "Odd vs Even Prediction Distribution", ["Even", "Odd"])
    np.savetxt("m1_predictions.txt", y_odd_even, fmt="%d")
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title("Model 1 - Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("m1_loss_plot.png")
    plt.show()
    correct = conf_matrix.diagonal()
    total = conf_matrix.sum(axis=1)
    accuracy_per_class = correct / total

    brp.drawBarPlot(accuracy_per_class, list(range(10)), title="Model 1 - Accuracy Per Class")


if __name__ == '__main__':
    main()