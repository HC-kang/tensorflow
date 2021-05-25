from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# input이 1개인 선형회귀
training_data = np.array([[1.], [2.], [3.], [4.],[5.]])
target_data = np.array([[3.], [5.],[7.],[9,],[11.]])

Inputs = keras.Input(shape = (1))
outputs = layers.Dense(1, activation = 'linear')(Inputs)
model = keras.Model(inputs = Inputs, outputs = outputs, name = 'linear')
model.compile(loss = 'mse', optimizer = 'sgd', metrics = ['accuracy'])
model.fit(training_data, target_data, epochs = 100, verbose = 1)
print(model.summary())
print(model.layers[1].get_weights())

inp = list(map(int, input().split()))
inp = np.array(inp)
print('입력 : {}'.format(inp))
print('결과값 : {}'.format(model.predict(inp)[0].round()))

# Input 이 2개인 선형회귀

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

training_data = np.array([[1., 2.], [2.,3.], [3.,4.],[4.,5.],[5.,6.]])
target_data = np.array([[3.],[5.],[7.],[9.],[11.]])

Inputs = keras.Input(shape = (2))
outputs = layers.Dense(1, activation = 'linear')(Inputs)
model = keras.Model(inputs = Inputs, outputs = outputs, name = 'linear')
model.compile(loss = 'mse', optimizer = 'sgd', metrics = 'accuracy')
model.fit(training_data, target_data, epochs = 100, verbose = 1)
print(model.summary())
print(model.layers[1].get_weights())

inp = list(map(int, input().split()))
inp = np.array(inp)
print('입력 : {}'.format(inp))
inp = inp.reshape(1,2)
print('결과값 : {}'.format(model.predict(inp)[0].round()))



# input 2개, output 2개인 선형회귀

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

training_data = np.array([[1.,2.],[2.,3.],[3.,4.],[4.,5.],[5.,6.]])
target_data = np.array([[3.,2.],[5.,6.],[7.,12.],[9.,20.],[11.,30.]])

Inputs = keras.Input(shape = (2))
outputs = layers.Dense(2, activation = 'linear')(Inputs)
model = keras.Model(inputs = Inputs, outputs = outputs, name = 'linear2d')
model.compile(loss = 'mse', optimizer = 'sgd', metrics = 'accuracy')
model.fit(training_data, target_data, epochs = 100, verbose = 3)
print(model.summary())
print(model.layers[1].get_weights())

inp = list(map(int, input().split()))
inp = np.array(inp)
print('입력 : {}'.format(inp))
inp = inp.reshape(1,2)
print('결과값 : {}'.format(model.predict(inp)[0].round()))


# 은닉층 2개인 딥러닝 구현
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

training_data = np.array([[1.,2.],[2.,3.],[3.,4.],[4.,5.],[5.,6.]])
target_data = np.array([[3.,2.],[5.,6.],[7.,12.],[9.,20.],[11.,30.]])

inputs = keras.Input(shape = (2))
x1 = layers.Dense(4, activation = 'sigmoid')(inputs)
x2 = layers.Dense(4, activation = 'sigmoid')(x1)
outputs = layers.Dense(2, activation = 'linear')(x2)
model = keras.Model(inputs = inputs, outputs = outputs, name = 'linear2D')
model.compile(loss = 'mse', optimizer = 'sgd', metrics = 'accuracy')
model.fit(training_data, target_data, epochs = 100, verbose = 1)
print(model.summary())
print(model.layers[1].get_weights())

inp = list(map(int, input().split()))
inp = np.array(inp)
print('입력 : {}'.format(inp))
inp = inp.reshape(1,2)
print('결과값 : {}'.format(model.predict(inp)[0].round()))