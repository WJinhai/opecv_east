from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
import os
from PIL import Image
import numpy as np

def load_data():
    data = np.empty((100000,1,28,28),dtype="float32")
    label = np.empty((100000,),dtype="uint8")

    imgs = os.listdir("E:\\image_python\\data\\Minst\\min_train")
    num = len(imgs)
    for i in range(num):
        img = Image.open("E:\\image_python\\data\\Minst\\min_train\\"+imgs[i])
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
        label[i] = int(imgs[i].split('.')[0])
    data = data.reshape(100000,28,28,1)
    index = [i for i in range(len(data))]
    np.random.shuffle(index)
    data = data[index]
    label = label[index]

    return data,label

data , label = load_data()

print(data.shape[0], ' samples')

label = np_utils.to_categorical(label, 10)
train_data = data[:80000]
train_labels = label[:80000]

validation_labels = label[20000:]
validation_data = data[20000:]

model = Sequential()

model.add(Convolution2D(4, 5, 5,input_shape=(28, 28,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(8, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_data, train_labels,
          nb_epoch=10, batch_size=100,
          validation_data=(validation_data, validation_labels))


json_string = model.to_json()
open('D:\\image_python\\data\\Minst\\my_model.json','w').write(json_string)
model.save_weights('D:\\image_python\\data\\Minst\\model.h5')