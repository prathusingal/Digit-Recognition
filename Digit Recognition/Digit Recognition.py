import keras
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape
x_test.shape
y_train.shape
y_test.shape
import matplotlib.pyplot as plt
def plot_image(img):
  #(784,) => (28,28)
  img = img.reshape(28,28)
  plt.imshow(img, cmap="gray")
plot_image(x_train[456])
print(y_train[456])
x_train = x_train.reshape(60000,784 )
x_test = x_test.reshape(10000, 784)
print(x_train.shape)
print(x_test.shape)
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)
print(y_test.shape)
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add( Dense(units = 32, activation='relu', input_shape = (784, )) ) # input layer

model.add( Dense(units = 64, activation='relu' ) )  # no need to give input_shape
model.add( Dense(units = 128, activation='relu' ) )  # no need to give input_shape
model.add( Dense(units = 32, activation='relu', ) )  # no need to give input_shape

model.add( Dense(units = 10, activation='softmax' ) )  # output layer 
model.compile(optimizer= "adam", loss="categorical_crossentropy", metrics=['accuracy'])
# train our model
model.fit(x = x_train, y= y_train, epochs= 10, validation_data=(x_test, y_test))
plot_image( x_test[9000] )
model.predict_classes(x_test[9000].reshape(1, 784))