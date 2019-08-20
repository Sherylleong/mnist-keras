# Every line of these files consists of an image, i.e. 785 numbers between 0 and 255.  size 28 x 28
# first no is label
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_rows, img_cols = 28, 28

#plot the first image in the dataset
# plt.imshow(x_train[0])



# reshape for channel last
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

#one-hot encoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#create model
model = Sequential()

# layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape)) # kernel size is filter matrix size
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax')) # softmax makes output sum up to 1


# compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# train the model
seqModel = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)

# evaluate model on test set
model.evaluate(x_train,y_train,batch_size=128) # returns loss and accuracy

# predict first 10 images in the test set
print(model.predict(x_test[:10]))

# actual results for first 10 images in test set
print(y_test[:10])

# save model as hdf5 file
from tensorflow.keras.models import load_model
model.save('mnist.h5')
