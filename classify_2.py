# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:04:44 2019

@author: laura
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:46:54 2019

@author: laura
"""

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from load_images import load_data
import os

IMG_SIZE = 128

aug = ImageDataGenerator(horizontal_flip=True)

# Load data
if os.path.exists("X_color_128.npy") and os.path.exists("y_color_128.npy"):
    X = np.load("X_color_128.npy")
    y = np.load("y_color_128.npy")
else:
    X, y = load_data(IMG_SIZE)

print("X shape is: ", X.shape)
X, y = shuffle(X,y)
X = X[:4000, :,:,:] 
y = y[:4000]

#Normalizing the data to scale 0-1
print("Max on: ", X.max())
X = X / X.max()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)

#create model

base_model = MobileNet(input_shape=(128,128,3), include_top=False, weights='imagenet')

x = GlobalAveragePooling2D() (base_model.output)
x = Dense(512, activation='relu') (x)
x = Dropout(0.2) (x)
x = Dense(2, activation='softmax') (x)

model = Model(inputs=base_model.inputs, outputs=x)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit_generator(aug.flow(X_train, y_train, batch_size=32), epochs=10, steps_per_epoch=10, validation_steps=1,validation_data=(X_test, y_test))