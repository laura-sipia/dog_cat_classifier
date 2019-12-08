# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:46:54 2019

@author: laura
"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from load_images import load_data
import os

IMG_SIZE = 64
DATA_PATH="../data/train"
BATCH_SIZE=32
#%%
#create model
# Best test acc score 0.716 with 4000 samples where 0.2 are test samples
# For 64 feature maps
# 10/10 ---- loss: 0.5781 - acc: 0.7666 - val_loss: 0.6023 - val_acc: 0.7125
model = Sequential()

N_1 = 16 # Number of feature maps
N_2 = 32 
N_3 = 64 
w, h = 3, 3 # Conv. window size

model.add(Conv2D(N_1, (w, h), 
                 input_shape=(IMG_SIZE, IMG_SIZE, 3),
                 activation = 'relu',
                 padding = 'same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(N_2, (w, h),
                 activation = 'relu',
                 padding = 'same'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(N_3, (w, h),
                 activation = 'relu',
                 padding = 'same'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(N_3, (w, h),
                 activation = 'relu',
                 padding = 'same'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(GlobalAveragePooling2D())

model.add(Dense(1024, activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(512, activation = 'relu'))

model.add(Dropout(0.4))

model.add(Dense(2, activation = 'softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#%%

aug_gen = ImageDataGenerator(
        horizontal_flip=True,
        samplewise_std_normalization=True,
        validation_split=0.2
        )

train_gen = aug_gen.flow_from_directory(
        directory=DATA_PATH,
        target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        subset="training" 
        )

validation_gen = aug_gen.flow_from_directory(
        directory=DATA_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        subset="validation"
        )
#%%
#NOT USED
"""
if os.path.exists("X_color_64.npy") and os.path.exists("y_color_64.npy"):
    X = np.load("X_color_64.npy")
    y = np.load("y_color_64.npy")
else:
    X, y = load_data(IMG_SIZE)

print("X shape is: ", X.shape)
X, y = shuffle(X,y)
#X = X[:4000, :,:,:] 
#y = y[:4000]

#Normalizing the data to scale 0-1
print("Max on: ", X.max())
X = X / X.max()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
"""
#%%
# Fit with auto-generated data
model.fit_generator(
        train_gen,
        steps_per_epoch=train_gen.samples//BATCH_SIZE,
        validation_steps=validation_gen.samples//BATCH_SIZE,
        validation_data=validation_gen,
        epochs=15
        )
model.save("custom_model_1.h5")
#%%
#NOT USED
"""
model.fit_generator(aug_gen.flow(X_train, y_train, batch_size=32), epochs=10, steps_per_epoch=25000//32, validation_steps=1,validation_data=(X_test, y_test))
"""