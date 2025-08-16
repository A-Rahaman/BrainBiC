import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import os

class BiclusterDataset(Dataset):
    """
    This dataset creates training batches with appropriate input and reference 
    @rahaman
    """
    
    def __init__(self, data, transform=None):
        self.data = torch.FloatTensor(data)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, sample  # Return same sample as input and target for autoencoder

class DataManager:
    """
    Handles data loading, preprocessing, and dataloader creation
    Helps efficeint data sampling
    """
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
    def load_data(self, data_path=None):
        """Load data from file or generate synthetic data"""
        if data_path and os.path.exists(data_path):
            # Load from file
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
                data = df.values
            elif data_path.endswith('.parquet'):
                data = pd.read_parquet('your_file.parquet')
            elif data_path.endswith('.npy'):
                data = np.load(data_path)
            else:
                raise ValueError("Unsupported file format. Use .csv, .parquet or .npy")
        else:
            # Generate synthetic data for demonstration
            print("Generating synthetic data...")
            data, _ = make_classification(
                n_samples=self.config['n_samples'],
                n_features=self.config['n_features'],
                n_informative=self.config['n_informative'],
                n_redundant=self.config['n_redundant'],
                n_clusters_per_class=self.config['n_clusters_per_class'],
                random_state=self.config['random_state']
            )
        
        return data
    
    def preprocess_data(self, data):
        """
        If necessary: standardize the data
        """
        if self.config['normalize']:
            data = self.scaler.fit_transform(data)
        return data
    
    def create_dataloaders(self, data_path=None):
        """Create train and test dataloaders"""
        # Load and preprocess data
        data = self.load_data(data_path)
        data = self.preprocess_data(data)
        
        # Split data
        n_train = int(len(data) * self.config['train_ratio'])
        train_data = data[:n_train]
        test_data = data[n_train:]
        
        # Create datasets
        train_dataset = BiclusterDataset(train_data)
        test_dataset = BiclusterDataset(test_data)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        return train_loader, test_loader, data