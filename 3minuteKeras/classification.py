from keras import layers, models


# 분상형 지정 방식
x = layers.Input(shape=(Nin,))
h = layers.Activatoin('relu')(layers.Dense(Nh)(x))
y = layers.Activation('softmax')(layers.Dense(Nout)(h))

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])


# 연쇄형 지정 방식
model = models.Sequential()

model.add(layers.Dense(Nh, activation='relu', input_shaep=(Nin,))) # 입력과 은닉이 동시에 정해짐
model.add(layers.Dense(Nout, activation='softmax'))

# 객체 지향형 구현
class ANN(models.Model):
    def __init__(self, Nin, Nh, Nout):

hidden = layers.Dense(Nh)

import numpy as np
from keras import datasets #mnist
from keras.utils import np_utils #to_categorical

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

Y_train = np_utils.to_categorical(y_train) # 0~9 정수를 이진 벡터로 변환, 효율적이라서
Y_test = np_utils.to_categorical(y_test)

L, W, H = X_train.shape
X_train = X_train.reshape(-1, W*H) # L개의 이미지를 저장 ( L * W * H )
X_test = X_test.reshape(-1, W*H)

X_train = X_train/ 255.0 # 각 이미지의 구성된 값을 0~1사이의 값으로 변환
X_test = X_test / 255.0


import matplotlib.pyplot as plt

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)

def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)

def main():
    Nin = 784
    Nh = 100
    number_of_class = 10
    Nout = number_of_class