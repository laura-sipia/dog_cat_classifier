# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:04:44 2019

@author: laura
"""
#%%
import os
import tensorflow
#import imgaug as iaa
import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from load_images import load_data, load_test_images


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



#%%
IMG_SIZE = 128
TRAIN_PATH = "../data/train"
DATA_PATH = "../data"
TEST_PATH = "../data/test/"
BATCH_SIZE = 32

base_gen = ImageDataGenerator(horizontal_flip=True,validation_split=0.2)
test_base_gen = ImageDataGenerator()

#%%

# Loading and resizing data by itself not used
"""

# Load data
if os.path.exists("X_color_128.npy") and os.path.exists("y_color_128.npy"):
    X = np.load("X_color_128.npy")
    y = np.load("y_color_128.npy")
else:
    X, y = load_data(IMG_SIZE)

print("X shape is: ", X.shape)
X, y = shuffle(X,y)
X = X[:10000, :,:,:] 
y = y[:10000]

#Normalizing the data to scale 0-1
print("Max on: ", X.max())
X = X / X.max()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
"""
#%%
# Using image generator for data input

train_gen = base_gen.flow_from_directory(
        directory=TRAIN_PATH,
        target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        subset="training" 
        )

validation_gen = base_gen.flow_from_directory(
        directory=TRAIN_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        subset="validation"
        )
#%%
# Image augmentation
# NOT USED
"""
seq = iaa.Sequential([
        iaa.Fliplr(0.5)
        ])
    
def augment_gen():
    while True:
        batch = next(train_gen)
        yield (seq.augment_images(batch[0], batch[1]))

aug_gen = augment_gen()
"""
#%%
#create model

base_model = MobileNet(input_shape=(128,128,3), include_top=False, weights='imagenet')

x = GlobalAveragePooling2D() (base_model.output)
#x = Dense(512, activation='relu') (x)
x = Dropout(0.4) (x)
x = Dense(2, activation='softmax') (x)

model = Model(inputs=base_model.inputs, outputs=x)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy']
              )
sv_checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(
        "model_mobilenet_checkpoint.h5",
        monitor='val_loss',
        save_best_only=True
        )
#%%

# Fit with auto-generated data
model.fit_generator(
        train_gen,
        steps_per_epoch=train_gen.samples//BATCH_SIZE,
        validation_steps=validation_gen.samples//BATCH_SIZE,
        validation_data=validation_gen,
        epochs=5
        )
model.save("mobilenet_model.h5")

#%%
# Fit with "manually" generated data
# NOT USED
"""
model.fit_generator(base_gen.flow(X_train, y_train, batch_size=32), epochs=10, steps_per_epoch=10, validation_steps=1,validation_data=(X_test, y_test))
"""
#%%
# Predict lables of images
if model == None:
    model = load_model("mobilenet_model.h5")
    model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy']
              )

images = load_test_images(IMG_SIZE, TEST_PATH)
predictions = model.predict(images, batch_size=10)
predicted_class = np.argmax(predictions,axis=1)
print(predicted_class)
#%%

# Save test images with corresponding label to file

i = 0
for filename in os.listdir(TEST_PATH):
    label = ""
    if predicted_class[i] == 0:
        label = "cat"
    else:
        label = "dog"
    new_name = label+ str(i) + ".jpg"
    src = TEST_PATH + filename
    new_name = TEST_PATH + new_name 
    
    os.rename(src, new_name)
    i += 1
