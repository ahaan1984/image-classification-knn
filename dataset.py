import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2

class Dataset:
    def __init__(self, path:str, width:int, height:int, batch_size:int) -> None:
        self.path = path
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.images = []

    def _load(self, path: str):
        img = cv2.imread(path)
        return img
    
    def load_images(self): # -> NDArray
        images = []
        for filename in os.listdir(self.path):
            if filename.endswith('jpg') or filename.endswith('png'):
                img_path = os.path.join(self.path, filename)
                img = self._load(img_path)
                img = cv2.resize(img, (self.width, self.height))
                images.append(img)

        return np.array(images)
    
    def split_dataset(self, train_split, val_split, test_split):
        images = self.load_images()
        train_split /= 100
        val_split /= 100
        test_split /= 100
        # if test_size < 0:
        #     raise ValueError("The sum of train_size and valid_size should be less than or equal to 1."
        # Calculate the number of images for each set
        num_images = len(images)
        num_train = int(num_images * train_split)
        num_valid = int(num_images * val_split)

        # Split the data into training, validation and test sets
        trainset = images[:num_train]
        valset = images[num_train:num_train + num_valid]
        testset = images[num_train + num_valid:]

        return trainset, valset, testset
    
    def create_batches(self, train_size:int, valid_size:int, test_split:int, shuffle:bool):
        train_images, valid_images, test_images = self.split_dataset(train_size, valid_size, test_split)

        if shuffle == True:
            np.random.shuffle(train_images)
            # np.random.shuffle(valid_images)
            # np.random.shuffle(test_images)

        train_batches = [train_images[i:i + self.batch_size] for i in range(0, len(train_images), self.batch_size)]
        valid_batches = [valid_images[i:i + self.batch_size] for i in range(0, len(valid_images), self.batch_size)]
        test_batches = [test_images[i:i + self.batch_size] for i in range(0, len(test_images), self.batch_size)]

        return train_batches, valid_batches, test_batches
    


    