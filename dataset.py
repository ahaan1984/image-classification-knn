import os
import numpy as np
import pandas as pd
from typing import Union
import cv2 as cv2

class Dataset:
    def __init__(self, path:str, width:int, height:int) -> None:
        self.path = path
        self.width = width
        self.height = height
    
    def load_images(self, label_path, dimension=(64,64)):
        images = []
        for filename in os.listdir(label_path):
            img = cv2.imread(os.path.join(label_path, filename))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
                img_normal = img_resized / 255
                images.append(img_normal)
        return np.array(images)
    
    def load_labels(self, label1, label2):
        y0 = np.zeros(label1)
        y1 = np.ones(label2)
        y = []
        y = np.concatenate((y0, y1), axis=0)
        return np.array(y)
    
    
    
  

    
    