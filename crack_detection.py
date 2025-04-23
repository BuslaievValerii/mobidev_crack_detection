import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras import Input
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, BatchNormalization, Dense, Flatten

num_anomaly = len(os.listdir("train/anomaly"))
num_normal  = len(os.listdir("train/normal"))

print(f"There are {num_anomaly} photos with anomalies and {num_normal} without them")

X, Y = [], []
for f in os.listdir("train/anomaly"):
    img = cv2.imread(f"train/anomaly/{f}")
    X.append(img)
    Y.append(1)

normal_full = os.listdir("train/normal")
indices = np.random.choice(len(normal_full), size=num_anomaly*3)
normal_sample = [normal_full[i] for i in indices]

for f in normal_sample:
    img = cv2.imread(f"train/normal/{f}")
    X.append(img)
    Y.append(0)  
Y = to_categorical(Y)

X, Y = np.array(X), np.array(Y)

IMG_ROWS, IMG_COLS = 227, 227
BATCH_SIZE = 32
NUM_EPOCHS = 10

def build_lenet():
    model = Sequential()

    model.add(Input(shape=(IMG_ROWS, IMG_COLS, 3)))
    model.add(Conv2D(6, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    
    model.add(Flatten()) 
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(2, activation = 'softmax'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'precision', 'recall', 'f1_score'])
    return model

lenet = build_lenet()
lenet.fit(X, Y, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_split=0.25, shuffle=True)
lenet.save("lenet.keras")

def load_test_data(path):
    X, Y = [], []
    for f in os.listdir(f"{path}/anomaly"):
        img = cv2.imread(f"{path}/anomaly/{f}")
        X.append(img)
        Y.append(1)
    
    for f in os.listdir(f"{path}/normal"):
        img = cv2.imread(f"{path}/normal/{f}")
        X.append(img)
        Y.append(0)  
    Y = to_categorical(Y)
    
    X, Y = np.array(X), np.array(Y)
    return X, Y

def visualize_result(actual, predicted):
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["normal", "anomaly"])
    cm_display.plot()
    plt.show()

    accuracy = metrics.accuracy_score(actual, predicted)
    precision = metrics.precision_score(actual, predicted)
    recall = metrics.recall_score(actual, predicted)
    f1 = metrics.f1_score(actual, predicted)

    print(f"Accuracy:  {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall:    {recall}")
    print(f"F1 score:  {f1}")

X_valid, Y_valid = load_test_data("valid_balanced")
if 'lenet' in globals():
    Y_predicted = lenet.predict(X_valid)
elif "lenet.keras" in os.listdir():
    lenet = load_model("lenet.keras")
    Y_predicted = lenet.predict(X_valid)
else:
    print("Ther is no model neither trained nor loaded")

if "Y_predicted" in globals():
    visualize_result(np.argmax(Y_valid, axis = 1), np.argmax(Y_predicted, axis = 1))

X_valid, Y_valid = load_test_data("valid_unbalanced")
if 'lenet' in globals():
    Y_predicted = lenet.predict(X_valid)
elif "lenet.keras" in os.listdir():
    lenet = load_model("lenet.keras")
    Y_predicted = lenet.predict(X_valid)
else:
    print("Ther is no model neither trained nor loaded")
if "Y_predicted" in globals():
    visualize_result(np.argmax(Y_valid, axis = 1), np.argmax(Y_predicted, axis = 1))
