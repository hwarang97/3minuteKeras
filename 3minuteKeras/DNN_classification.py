# 기본 파라미터 설정
Nin = 784
Nh_1 = [100, 50]
number_of_class = 10
Nout = number_of_class

# 분류 DNN 모델 구현
from keras import layers, models
class DNN(models.Sequential):
    def __init__(self, Nin, Nh_1, Nout):
        super().__init__()
        self.add(layers.Dense(Nh_1[0], activation='relu', input_shape=(Nin,), name='Hidden-1'))
        self.add(layers.Dropout(0.2))
        self.add(layers.Dense(Nh_1[1], activation='relu', name='Hidden-2')) # input_shape가 자동적으로 설정된다
        self.add(layers.Dropout(0.2))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 데이터 준비
import numpy as np
from keras import datasets
import tensorflow as tf
from keras.utils import np_utils
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
y_train = np_utils.to_categorical(y_train) # 0~9 정수를 이진벡터로 변환. 효율적이라서
y_test = np_utils.to_categorical(y_test)
L, W, H = X_train.shape
X_train = X_train.reshape(-1, W * H)
X_test = X_test.reshape(-1, W * H)
X_train = X_train / 255.0
X_test = X_test / 255.0

# 분류 DNN 학습 및 성능 평가
model = DNN(Nin, Nh_1, Nout)
history = model.fit(X_train, y_train, epochs=10, batch_size=100, validation_split=0.2)
performance_test = model.evaluate(X_test, y_test, batch_size=100)
print('Test Loss and Accuracy ->', performance_test)