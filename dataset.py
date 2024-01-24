import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv2

class Dataset:
    def __init__(self, path:str, label_path:str, width:int, height:int, batch_size:int) -> None:
        self.path = path
        self.label_path = label_path
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.images = []
        # self.image_labels_map = self.load_labels()

    def _load(self, path: str):
        img = cv2.imread(path)
        return img
    
    def load_labels(self):
        labels_df = pd.read_csv(self.label_path)
        labels = dict(zip(labels_df['image'], labels_df['labels']))
        return list(labels.values())
    
    def load_images(self): # -> NDArray
        images = []
        # labels = []
        for filename in os.listdir(self.path):
            if filename.endswith('jpg') or filename.endswith('png'):
                img_path = os.path.join(self.path, filename)
                img = self._load(img_path)
                img = cv2.resize(img, (self.width, self.height))
                images.append(img)
                # labels.append(self.image_labels_map.get(filename, -1))

        return images
    
    def preprocess_images(self, images):
        # Preprocess and normalize images
        processed_images = []
        for img in images:
            normalized_img = img / 255.0
            processed_images.append(normalized_img)
        return processed_images
    
    def split_dataset(self, train_split, val_split, test_split):
        images = self.load_images()
        labels = self.load_labels()
        images= self.preprocess_images(images)
        images, labels = np.array(images, dtype='float16'), np.array(labels, dtype='float16')
        
        idx = np.arange(len(images))
        np.random.shuffle(idx)
        images, labels = images[idx], labels[idx]

        train_split /= 100
        val_split /= 100
        test_split /= 100

        num_images = len(images)
        num_train = int(num_images * train_split)
        num_valid = int(num_images * val_split)

        # Split the data into training, validation and test sets
        trainimages = images[:num_train]
        trainlabels = labels[:num_train]
        valimages = images[num_train:num_train + num_valid]
        vallabels = images[num_train:num_train + num_valid]
        testimages = images[num_train + num_valid:]
        testlabels = images[num_train + num_valid:]

        trainset = (trainimages, trainlabels)
        valset = (valimages, vallabels)
        testset = (testimages, testlabels)

        return trainset, valset, testset
    
    def create_batches(self, dataset, batch_size):
        images, labels = dataset
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            yield batch_images, batch_labels
    
    