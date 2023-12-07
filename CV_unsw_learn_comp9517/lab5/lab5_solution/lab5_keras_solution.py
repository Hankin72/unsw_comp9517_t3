import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D

# Load data(do not change)
data = pd.read_csv("src/mnist_train.csv")
train_data = data[:2000]
test_data = data[2000:2500]

train_label = train_data['label']
test_label = test_data['label']

train_label = np_utils.to_categorical(train_label, 10)
test_label = np_utils.to_categorical(test_label, 10)

train_data = train_data.drop(["label"], axis = 1)
train_data = train_data.to_numpy() / 255
train_data = train_data.reshape(-1, 28, 28, 1)

test_data = test_data.drop(["label"], axis = 1)
test_data = test_data.to_numpy() / 255
test_data = test_data.reshape(-1, 28, 28, 1)

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

def PlotLearningCurve(epoch, trainingloss, testingloss):
    plt.plot(range(epoch), trainingloss, 'b', range(epoch), testingloss, 'r')
    plt.title('learning curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training loss', 'testing loss'])
    plt.show()
    
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_data, train_label, batch_size = 2000, epochs = 50, verbose = 1, validation_data=(test_data, test_label))

PlotLearningCurve(50, history.history['loss'], history.history['val_loss'])
    
score = model.evaluate(test_data, test_label, verbose=0)
print("final test accuracy", score)

