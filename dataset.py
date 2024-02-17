import os
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2

class Dataset:
    __slots__ = ['path', 'width', 'height']

    def __init__(self, path:str, width:int, height:int) -> None:
        self.path = path
        self.width = width
        self.height = height
    
    def load_images(self, path:str) -> np.ndarray:
        images = []
        dimension=(self.width, self.height)
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
                img_normal = (img_resized - np.mean(img_resized)) / np.std(img_resized)
                images.append(img_normal)
        return np.array(images)
    
    def load_labels(self, label1:int, label2:int) -> np.ndarray:
        y0 = np.zeros(label1)
        y1 = np.ones(label2)
        y = []
        y = np.concatenate((y0, y1), axis=0)
        return np.array(y)

    
    def plot_random_samples(self, images, num_samples):
        images = list(images)
        num_samples = min(num_samples, len(images))
        samples = random.sample(images, num_samples)
        plt.figure(figsize=(20, 20))
        for i, sample in enumerate(samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(sample, cmap='gray')
            plt.axis('off')
        plt.show()
    
  

    
    