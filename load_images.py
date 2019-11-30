# -*- coding: utf-8 -*-

import glob
import numpy as np
import os
from tqdm import tqdm 
import cv2

IMG_SIZE=64

def load_data():
    X = []          # Images go here
    y = []          # Class labels go here
    classes = []    # All class names go here
    # Find all files from this folder
    files = glob.glob("./data/train" + os.sep + "*.jpg")
    # Load all files
    for name in tqdm(files):
        # Load image and parse class name
        img = cv2.imread(name, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
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
    
    np.save("X.npy", X)
    np.save("y.npy", y)
    
    return X, y