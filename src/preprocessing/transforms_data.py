from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
import torch
import pandas as pd


class data_transformations():

    def __init__(self, train_dir, test_dir, model_transforms, BATCH_SIZE=32):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.model_transforms = model_transforms
        self.BATCH_SIZE = BATCH_SIZE
        self.train_data = datasets.ImageFolder(root=self.train_dir,
                                        transform=self.model_transforms)
        self.test_data = datasets.ImageFolder(root=self.test_dir,
                                        transform=self.model_transforms)
        
        train_size = int(0.9*len(self.train_data))
        valid_size = len(self.train_data) - train_size
        self.train_data, self.valid_data = random_split(self.train_data, [train_size, valid_size])

    def get_dataloaders(self):
        '''
        This function is used to get the three dataloaders: train_dataloader, valid_dataloader, test_dataloader
        The data from train_dir is split into 'train_data' and 'valid_data'
        The test_dir is used to generate the test_data.
        '''

        train_dataloader = DataLoader(dataset=self.train_data,
                                batch_size=self.BATCH_SIZE,
                                shuffle=True)
        valid_dataloader = DataLoader(dataset=self.valid_data,
                                    batch_size=self.BATCH_SIZE)
        test_dataloader = DataLoader(dataset=self.test_data,
                                    batch_size=self.BATCH_SIZE)
        
        
        return train_dataloader, valid_dataloader, test_dataloader
    
    def get_class_weights_inverse_train(self):
        """Calculate weights as inverse of class frequency"""
        total_train_samples = len(self.train_data)
        class_counts = pd.Series([k[1] for k in self.train_data]).value_counts().to_dict()
        weights = []
        for class_id in sorted(class_counts.keys()):
            weight = total_train_samples / (len(class_counts) * class_counts[class_id])
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float32)