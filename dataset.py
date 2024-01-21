import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2

class Dataset:
    def __init__(self, path, width, height, batch_size):
        self.path = path
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.images = []

    def load_images(self):
        for filename in os.listdir(self.path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img = cv2.imread(os.path.join(self.path, filename))
                if img is not None:
                    img = cv2.resize(img, (self.width, self.height))
                    self.images.append(img)

    # def load_batches(self):
    #     for i in range(0, len(self.images), self.batch_size):
    #         yield self.images[i:i + self.batch_size]

    