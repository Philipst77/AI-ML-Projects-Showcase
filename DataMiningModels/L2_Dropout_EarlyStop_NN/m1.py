import numpy as np
import tensorflow as tf
import ConvertToOddEven as ote
import matplotlib.pyplot as plt
import seaborn as sns
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
    history = model1.fit(X_train, y_train, epochs  = 250, validation_split=0.2, verbose=1 )
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title("Model 1 - Loss Curve")  
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("m1_loss_curve.png")  
        plt.close()
    model1.save(parms.outModelFile)
    X_val = np.load("MNIST_X_test_1.npy")
    y_val = np.load("MNIST_y_test_1.npy")
    (X_val, y_val) = processTestData(X_val, y_val)
    y_pred_probs = model1.predict(X_val) 
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    y_odd_even = ote.convertToOddEven(y_pred_classes)
    np.savetxt("m1_predictions.txt", y_odd_even, fmt="%d")
    
    
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.title("Model 1 - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("m1_confusion_matrix.png")
    plt.close()

    correct = conf_matrix.diagonal()
    total = conf_matrix.sum(axis=1)
    accuracy_per_class = correct / total
    plt.figure(figsize=(8, 6))
    plt.bar(range(10), accuracy_per_class)
    plt.title("Model 1 - Accuracy Per Class")
    plt.xlabel("Digit Class")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.savefig("m1_accuracy_per_class.png")
    plt.close()

        
    X_autolab = np.load("MNIST_autolabTest_X.npy")
    X_autolab = X_autolab.reshape(X_autolab.shape[0], 784) / 255.0
    y_autolab_pred = model1.predict(X_autolab)
    y_autolab_classes = np.argmax(y_autolab_pred, axis=1)
    y_autolab_binary = ote.convertToOddEven(y_autolab_classes)
    np.savetxt("m1_final_predictions.txt", y_autolab_binary, fmt="%d")




    
if __name__ == '__main__':
    main()
 
