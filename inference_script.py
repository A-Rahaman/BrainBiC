import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm

# Import custom modules
from data_loader import DataManager
from model_definition import create_bicluster_model
from utils import ModelUtils

class BiclusterInference:
    
    def __init__(self, model_path, config_path=None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model checkpoint
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self.checkpoint.get('config', {})
        
        # Initialize model
        self.model = None
        self.input_dim = None
        
        # Storage for extracted features
        self.weights = {}
        self.activations = {}
        
    def setup_model(self, input_dim):
        self.input_dim = input_dim
        
        # Create model
        self.model = create_bicluster_model(
            model_type=self.config.get('model_type', 'simple'),
            input_dim=input_dim,
            hidden_config=self.config.get('hidden_config', 5),
            dropout_rate=self.config.get('dropout_rate', 0.2)
        ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
    def get_weights(self):
        if hasattr(self.model, 'get_encoder_weights'):
            weights= self.model.get_encoder_weights().cpu().numpy()
        elif hasattr(self.model, 'get_bottleneck_weights'):  # Deep model
           weights = self.model.get_bottleneck_weights().cpu().numpy()
        
        return weights
        
    def extract_weights_activations(self, test_data):

        self.model.eval()
        with torch.no_grad():
            Z = self.model.get_hidden_activations(test_data)  # latent activations
            W = self.get_weights()    # weight matrix
        return W, Z

    def create_biclusters(self, data_loader, alpha=0.4, beta=0.7):
        
        # Setup model
        if self.model is None:
            # Get input dimension from first batch
            first_batch = next(iter(data_loader))
            input_dim = first_batch[0].shape[1]
            self.setup_model(input_dim)
        
        W,Z = self.extract_weights_activations(data_loader)
        
        # Feature assignments: |W_f,k| > alpha
        feature_assignments = torch.abs(W) > alpha
        
        # Subject assignments: |Z_s,k| > beta  
        subject_assignments = torch.abs(Z) > beta

        return feature_assignments, subject_assignments

    

'''
data_manager = DataManager(data_config)
data_loader, _, _ = data_manager.create_dataloaders(args.data_path)

# Run inference
inference = BiclusterInference(args.model_path, args.config_path)
results = inference.create_biclusters(data_loader)
'''