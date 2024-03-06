import os
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2

class Dataset:
    __slots__ = ['path', 'width', 'height']

    def __init__(self, path: str, width: int, height: int) -> None:
        self.path = path
        self.width = width
        self.height = height

    def augment_image(self, img):
        # Rotation
        angle = np.random.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        tx = 0.2 * np.random.uniform() * img.shape[1]
        ty = 0.2 * np.random.uniform() * img.shape[0]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        if np.random.random() < 0.4:
            img = cv2.flip(img, 1)
        return img
    
    def load_images(self, path: str) -> np.ndarray:
        images = []
        dimension = (self.width, self.height)        
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
                img_normal = (img_resized - np.mean(img_resized)) / np.std(img_resized)
                img = cv2.GaussianBlur(img_normal, (3, 3), 0)
                if random.random() < 0.4:  # 50% chance to apply augmentations
                    img = self.augment_image(img)
                images.append(img)
        return np.array(images)
        
    def load_labels(self, label1: int, label2: int) -> np.ndarray:
        y0 = np.zeros(label1)
        y1 = np.ones(label2)
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
