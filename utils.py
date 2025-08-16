import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

class LossFunctions:

    # The constraints (loss) terms to be used across different experimental settings
    # Versions control
    
    @staticmethod
    def reconstruction_loss(reconstructed, target, loss_type='mse'):
        if loss_type == 'mse':
            return F.mse_loss(reconstructed, target)
        elif loss_type == 'bce':
            return F.binary_cross_entropy(reconstructed, target)
        else:
            raise ValueError("loss_type must be 'mse' or 'bce'")
    
    @staticmethod
    def sparsity_loss(W):
        return torch.sqrt(torch.sum(torch.abs(W)))
    '''
    def sparsity_loss(hidden_activations, sparsity_target=0.05):
        # Average activation across batch
        avg_activation = torch.mean(hidden_activations, dim=0)
        
        # KL divergence between target and actual sparsity
        kl_div = sparsity_target * torch.log(sparsity_target / (avg_activation + 1e-8)) + \
                 (1 - sparsity_target) * torch.log((1 - sparsity_target) / (1 - avg_activation + 1e-8))
        
        return torch.sum(kl_div)
    '''
    @staticmethod
    def semantic_locality_loss_glr(embeddings, adjacency_matrix, lambda_reg=1.0):
    
        # Compute semantic locality loss using weighted graph Laplacian regularization.
        
        # Weighted degree matrix
        degree = torch.sum(adjacency_matrix, dim=1)
        degree_matrix = torch.diag(degree)
        
        # Weighted Laplacian: L = D - W
        laplacian = degree_matrix - adjacency_matrix
        
        # Weighted Laplacian regularization: tr(X^T L X)
        loss = torch.trace(embeddings.T @ laplacian @ embeddings)
        
        return lambda_reg * loss
    
    @staticmethod
    def semantic_locality_loss(W, X, sigma=1.0):

        # Compute pairwise distances
        diff = X.unsqueeze(0) - X.unsqueeze(1)
        distances = torch.sum(diff ** 2, dim=2)

        # Similarity measure d_ij
        d_ij = torch.exp(-distances / sigma)
        
        # Weight differences |W_ki| - |W_kj|
        W_abs = torch.abs(W)
        weight_diff = W_abs.unsqueeze(2) - W_abs.unsqueeze(1)
    
    @staticmethod
    def biclustering_loss(W, Z):
        # optimizing the assignemnt operator (sample/features)

        P_feature = F.softmax(W, dim=1)
        freq = P_feature.sum(dim=0, keepdim=True)
        Q_feature = (P_feature**2 / freq) / (P_feature**2 / freq).sum(dim=1, keepdim=True)

        # Sample assignments H_sample
        H_sample = F.softmax(Z, dim=1)

        # Target distribution T_sample
        freq_sample = H_sample.sum(dim=0, keepdim=True)
        T_sample = (H_sample**2 / freq_sample) / (H_sample**2 / freq_sample).sum(dim=1, keepdim=True)

        # KL divergences
        kl_feature = F.kl_div(P_feature.log(), Q_feature, reduction='sum')
        kl_sample = F.kl_div(H_sample.log(), T_sample, reduction='sum')
        return kl_feature + kl_sample

    @staticmethod
    def orthogonality_loss(weight_matrix, ortho_weight=0.01):
        # Compute gram matrix
        gram = torch.mm(weight_matrix, weight_matrix.t())
        
        # Identity matrix
        identity = torch.eye(gram.size(0), device=weight_matrix.device)
        
        # Orthogonality loss
        ortho_loss = torch.norm(gram - identity, p='fro') ** 2
        
        return ortho_weight * ortho_loss
    
    @staticmethod
    def diversity_loss(hidden_activations, diversity_weight=0.1):
        
        # Diversity loss to encourage different neurons to activate for different inputs
        
        # Compute correlation matrix between neurons
        centered = hidden_activations - torch.mean(hidden_activations, dim=0)
        correlation = torch.mm(centered.t(), centered) / (hidden_activations.size(0) - 1)
        
        # Encourage low correlation between different neurons
        # Zero out diagonal elements
        mask = torch.eye(correlation.size(0), device=hidden_activations.device).bool()
        correlation = correlation[~mask]
        
        # Penalize high correlations
        diversity_loss = torch.mean(correlation ** 2)

        return diversity_weight * diversity_loss

class CompositeLoss:
    def __init__(self, loss_config):
        self.config = loss_config
        self.loss_functions = LossFunctions()
    
    def compute_loss(self, reconstructed, target, hidden_activations, encoder_weights,mu=0.5, delta=1.0, gamma=1.0, sigma=1.0):
        """
        Compute composite loss
        
        Args:
            reconstructed: Reconstructed output
            target: Original input
            hidden_activations: Hidden layer activations
            encoder_weights: Encoder weight matrix
        """
        total_loss = 0
        loss_components = {}
        
        # Reconstruction loss
        recon_loss = self.loss_functions.reconstruction_loss(
            reconstructed, target, self.config['reconstruction_loss_type']
        )
        total_loss += self.config['reconstruction_weight'] * recon_loss
        loss_components['reconstruction'] = recon_loss.item()
        
        # Sparsity loss
        if self.config['sparsity_weight'] > 0:
            sparsity_loss = self.loss_functions.sparsity_loss(
                hidden_activations, self.config['sparsity_target']
            )
            total_loss += self.config['sparsity_weight'] * sparsity_loss
            loss_components['sparsity'] = sparsity_loss.item()
        
        # Semantic locality loss
        if self.config['locality_weight'] > 0:
            locality_loss = self.loss_functions.semantic_locality_loss(
                hidden_activations, self.config['locality_weight']
            )
            total_loss += locality_loss
            loss_components['locality'] = locality_loss.item()
        
        # Assignment optimization loss
        if self.config['biclustering_weight'] > 0:
            bic_loss = self.loss_functions.biclustering_loss(
                hidden_activations, self.config['biclustering_weight']
            )
            total_loss += bic_loss
            loss_components['locality'] = bic_loss.item()

        # Orthogonality loss
        if self.config['orthogonality_weight'] > 0:
            ortho_loss = self.loss_functions.orthogonality_loss(
                encoder_weights, self.config['orthogonality_weight']
            )
            total_loss += ortho_loss
            loss_components['orthogonality'] = ortho_loss.item()
        
        # Diversity loss
        if self.config['diversity_weight'] > 0:
            div_loss = self.loss_functions.diversity_loss(
                hidden_activations, self.config['diversity_weight']
            )
            total_loss += div_loss
            loss_components['diversity'] = div_loss.item()
        
        loss_components['total'] = total_loss.item()

        # Orchestrated loss
        orc_loss = mu * (recon_loss) + (1 - mu) * (delta * bic_loss + gamma * sparsity_loss + locality_loss)
        
        return total_loss, loss_components, orc_loss

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.0001, restore_best_weights=True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        """
        Check if should stop training
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                self.restore_checkpoint(model)
            return True
        
        return False
    
    def save_checkpoint(self, model):
        """Save model weights"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
    
    def restore_checkpoint(self, model):
        """Restore best model weights"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

class ModelUtils:
    # utility functions for model operations
    # information about the model 
    
    @staticmethod
    def count_parameters(model):
        # Count total number of trainable parameters
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def save_model(model, path, additional_info=None):
        # Save model state dict and additional info
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_architecture': str(model),
        }
        
        if additional_info:
            save_dict.update(additional_info)
        
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    
    @staticmethod
    def load_model(model, path):
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
        return checkpoint
    
    @staticmethod
    def calculate_reconstruction_error(reconstructed, target):
        mse = F.mse_loss(reconstructed, target).item()
        mae = F.l1_loss(reconstructed, target).item()
        
        return {'mse': mse, 'mae': mae}

'''
class BiclusterMetrics:

    # a few cannonical evalaution metrics
    # in our use case, APCC, AEI, MSR 
    
    @staticmethod
    def calculate_bicluster_coherence(data, row_assignments, col_assignments):
        # Calculate coherence of biclusters
        coherence_scores = []
        
        for k in range(len(np.unique(row_assignments))):
            row_mask = (row_assignments == k)
            col_mask = (col_assignments == k)
            
            if np.sum(row_mask) > 0 and np.sum(col_mask) > 0:
                bicluster_data = data[row_mask][:, col_mask]
                # Calculate variance as coherence measure (lower is better)
                coherence = np.var(bicluster_data)
                coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else float('inf')
    
    @staticmethod
    def calculate_bicluster_coverage(row_assignments, col_assignments):
        # Calculate coverage of biclusters
        n_rows_covered = len(np.unique(row_assignments))
        n_cols_covered = len(np.unique(col_assignments))
        
        return {
            'row_coverage': n_rows_covered / len(row_assignments),
            'col_coverage': n_cols_covered / len(col_assignments)
        }
'''