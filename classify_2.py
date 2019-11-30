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

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from load_images import load_data
import os

#create model
# Best test acc score 0.716 with 4000 samples where 0.2 are test samples
model = Sequential()

N_1 = 64 # Number of feature maps in the first convolution
N_2 = 128 # Number of feature maps in the second convolution
w, h = 3, 3 # Conv. window size

model.add(Conv2D(N_1, (w, h), 
                 input_shape=(64, 64, 3),
                 activation = 'relu',
                 padding = 'same'))

model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(BatchNormalization())

model.add(Conv2D(N_2, (w, h),
                 activation = 'relu',
                 padding = 'same'))

model.add(GlobalAveragePooling2D())

model.add(BatchNormalization())

#model.add(Flatten())

model.add(Dense(512, activation = 'softmax'))

model.add(Dense(2, activation = 'softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

if os.path.exists("X.npy") and os.path.exists("y.npy"):
    X = np.load("X.npy")
    y = np.load("y.npy")
else:
    X, y = load_data()

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

model.fit(X_train, y_train, epochs=10, steps_per_epoch=10, validation_steps=1,validation_data=(X_test, y_test))