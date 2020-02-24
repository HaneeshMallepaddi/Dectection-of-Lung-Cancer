import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

batch_size = 128
num_classes = 10 
epochs = 2   #12  (Iterations)

img_rows, img_cols = 28, 28

x_train=np.load('C:\\Users\\Sai\\Desktop\\project\\x_train.npy')
x_test=np.load('C:\\Users\\Sai\\Desktop\\project\\x_test.npy')
y_train=np.load('C:\\Users\\Sai\\Desktop\\project\\y_train.npy')
y_test=np.load('C:\\Users\\Sai\\Desktop\\project\\y_test.npy')

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1) 

print('x_train shape:', x_train.shape) 
print(x_train.shape[0], 'train samples') 
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes) 
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential() 
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test)) 

score = model.evaluate(x_test, y_test, verbose=0) 

print('\n\tTest loss:', score[0])
print('\n\tTest accuracy:', score[1])
