from keras import layers, models

# 회귀 ANN 모델링
class ANN(models.Model):
    def __init__(self, Nin, Nh, Nout):
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('relu')

        x = layers.Input(shape=(Nin,)) # (Nin,)는 열벡터를 표현하는 방식이라고 한다.
        h = relu(hidden(x))
        y = output(h)

        super().__init__(x, y)
        self.compile(loss='mse', optimizer='sgd')

# 학습과 평가용 데이터 불러오기
from keras import datasets
from sklearn import preprocessing
import tensorflow as tf

def Data_func():

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()
    scaler = preprocessing.MinMaxScaler() # 최곳값을 1, 최저값을 0으로 정규화해주는 함수
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return (X_train, y_train), (X_test, y_test)

# 회귀 ANN 학습 결과 그래프 구현
from keraspp.skeras import plot_loss
import matplotlib.pyplot as plt

# 회귀 ANN 학습 및 성능 분석
def main():
    Nin = 13
    Nh = 5
    Nout = 1 # 분류가 아닌 회귀니까 값은 하나만 나오면 된다

    model = ANN(Nin, Nh, Nout)
    (X_train, y_train), (X_test, y_test) = Data_func()
    history = model.fit(X_train, y_train, epochs=100, batch_size=100, validation_split=0.2, verbose=2)

    performance_test = model.evaluate(X_test, y_test, batch_size=100) # 성능 평가시에는 사용하지 않은 데이터만 사용하는 것이 원칙
    print('\nTest Loss -> {:2f}'.format(performance_test)) # 결과가 검증때와 비슷한지 보고 비슷하다면 괜찮은 결과인것 같다.

    plot_loss(history)
    plt.show()

if __name__ =='__main__':
    main()