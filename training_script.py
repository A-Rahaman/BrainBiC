import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from datetime import datetime
import argparse
from tqdm import tqdm

# python3 -m pip install pandas

# Import custom modules (assuming they're in the same directory)
from data_loader import DataManager
from model_definition import create_bicluster_model
from utils import CompositeLoss, EarlyStopping, ModelUtils, DEFAULT_LOSS_CONFIG

class BiclusterTrainer:    
    def __init__(self, config):
        
        # Create a configuration file and use it to control the training process.
        # Easy to handle for different settings sicne the fucntions handles all diverse choices needed 
        # for our experiemnts to generate the performance   
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.early_stopping = None
        
        # Training history
        self.train_history = {'loss': [], 'epoch': []}
        self.val_history = {'loss': [], 'epoch': []}
        
    def setup_model(self, input_dim):
        self.model = create_bicluster_model(
            model_type=self.config['model_type'],
            input_dim=input_dim,
            hidden_config=self.config['hidden_config'],
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
    
    def setup_optimizer(self):
        if self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                momentum=0.9
            )
    
    def setup_loss_function(self):
        self.loss_fn = CompositeLoss(self.config['loss_config'])
    
    def setup_early_stopping(self):
        if self.config['early_stopping']['enabled']:
            self.early_stopping = EarlyStopping(
                patience=self.config['early_stopping']['patience'],
                min_delta=self.config['early_stopping']['min_delta'],
                restore_best_weights=self.config['early_stopping']['restore_best_weights']
            )
    
    def train_epoch(self, train_loader):
        # for a single epoch. this can be easily consumed for multiple epochs  
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed, hidden = self.model(data)
            
            # Get encoder weights
            if hasattr(self.model, 'get_encoder_weights'):
                encoder_weights = self.model.get_encoder_weights()
            else:
                encoder_weights = self.model.get_bottleneck_weights()
            
            # Calculate loss
            loss, loss_components, orc_loss = self.loss_fn.compute_loss(
                reconstructed, target, hidden, encoder_weights
            )
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            
            # Update progress bar: easy to monitor the training process 
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}"
            })
        
        # Average losses
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss
    
    def validate_epoch(self, val_loader):
        # for single epoch
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                reconstructed, hidden = self.model(data)
                
                # Get encoder weights
                if hasattr(self.model, 'get_encoder_weights'):
                    encoder_weights = self.model.get_encoder_weights()
                else:
                    encoder_weights = self.model.get_bottleneck_weights()
                
                # Calculate loss
                loss, loss_components, orc_loss = self.loss_fn.compute_loss(
                    reconstructed, target, hidden, encoder_weights
                )
                
                # Accumulate losses
                total_loss += loss.item()
        # Average losses
        avg_loss = total_loss / len(val_loader)
       
        return avg_loss
    
    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        # Saving the details of a traisned model 
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if specified
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved at epoch {epoch}")
    
    def train(self, train_loader, val_loader):
        # the trianer endpoint  
        print(f"Starting training on {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        # setup training components
        input_dim = next(iter(train_loader))[0].shape[1]
        self.setup_model(input_dim)
        self.setup_optimizer()
        self.setup_loss_function()
        self.setup_early_stopping()
        
        # training loop
        best_val_loss = float('inf')
        start_time = datetime.now()
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print("-" * 50)
            
            train_loss, train_components = self.train_epoch(train_loader)
            val_loss, val_components = self.validate_epoch(val_loader)
            
            # Log metrics
            self.train_history['loss'].append(train_loss)
            self.train_history['epoch'].append(epoch)
            self.val_history['loss'].append(val_loss)
            self.val_history['epoch'].append(epoch)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            self.save_checkpoint(epoch, train_loss, val_loss, is_best)
            
            # Early stopping check
            if self.early_stopping and self.early_stopping(val_loss, self.model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, 'final_model.pth')
        ModelUtils.save_model(
            self.model, 
            final_model_path,
            additional_info={
                'config': self.config,
                'best_val_loss': best_val_loss
            }
        )
        
        return self.model

'''
data_manager = DataManager(data_config)
train_loader, val_loader, _ = data_manager.create_dataloaders(args.data_path)

# Create trainer and start training
trainer = BiclusterTrainer(config)
trained_model = trainer.train(train_loader, val_loader)
'''