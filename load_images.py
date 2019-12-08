# -*- coding: utf-8 -*-

import glob
import numpy as np
import os
import cv2
from tqdm import tqdm 

def load_data(img_size):
    X = []          # Images go here
    y = []          # Class labels go here
    classes = []    # All class names go here
    # Find all files from this folder
    files = glob.glob("../data/train" + os.sep + "*.jpg")
    # Load all files
    for name in tqdm(files):
        # Load image and parse class name
        img = cv2.imread(name, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_size,img_size), interpolation=cv2.INTER_CUBIC)
        
        file_name = name.split(os.sep)[-1]
        class_name = file_name.split('.')[0]

        # Convert class names to integer indices:
        if class_name not in classes:
            classes.append(class_name)
        
        class_idx = classes.index(class_name)
        
        X.append(img)
        y.append(class_idx)
    # Convert python lists to contiguous numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    np.save("X_color_{}.npy".format(img_size), X)
    np.save("y_color_{}.npy".format(img_size), y)
    
    return X, y

def load_test_images(img_size, folder):
    images = []
    print(folder)
    files = glob.glob(folder + "*.jpg")
    for name in files:
        img = cv2.imread(name)
        img = cv2.resize(img, (img_size,img_size), interpolation=cv2.INTER_CUBIC)
        images.append(img)
    images = np.stack(images)
    return images