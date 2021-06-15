import numpy as np
from keras import datasets
from keras.utils import np_utils
import tensorflow as tf

def Data_func():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, W, H, C = X_train.shape
    X_train = X_train.reshape(-1, W * H *C) # 벡터화된 이미지를 여러개 저장장
   X_test = X_test.reshape(-1, W * H * C)


from keras import layers, models

class DNN(models.Sequential):
    def __init__(self, Nin, Nh_1, Pd_1, Nout):
        super().__init__()

        self.add(layers.Dense(Nh_1[0], activation='relu', input_shape=(Nin,), name='Hidden-1'))
        self.add(layers.Dropout(Pd_1[0]))
        self.add(layers.Dense(Nh_1[1], activation='relu', name='Hidden-1')) # 두번째부터는 알아서 입력 갯수를 계산
        self.add(layers.Dropout(Pd_1[1]))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


from keraspp.skeras import plot_loss, plot_acc
import matplotlib.pyplot as plt

