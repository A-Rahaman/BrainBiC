import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BiclusteringAutoencoder(nn.Module):
    """
    Biclustering Autoencoder with k hidden neurons (k = number of biclusters)
    Meta heuristic utilizes weight matrix and hidden activations for bicluster assignment
    """
    
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super(BiclusteringAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Assuming normalized input data
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Encode
        hidden = self.encoder(x)
        # Decode
        reconstructed = self.decoder(hidden)
        return reconstructed, hidden
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, hidden):
        return self.decoder(hidden)
    
    # For biclsuter assignemnet 
    def get_encoder_weights(self):
        return self.encoder[0].weight.data
    
    def get_decoder_weights(self):
        return self.decoder[0].weight.data
    
    def get_hidden_activations(self, x):
        with torch.no_grad():
            hidden = self.encode(x)
        return hidden

class BiclusteringAutoencoderDeep(nn.Module):
    """
    Deep Biclustering Autoencoder with multiple layers
    Bottleneck layer has k neurons for biclustering
    """
    
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super(BiclusteringAutoencoderDeep, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.bottleneck_dim = hidden_dims[-1]  # k (number of biclusters)
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:  # No dropout on bottleneck layer
                encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (symmetric to encoder)
        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims[:-1])) + [input_dim] # reconstructing back to the input dimension  
        prev_dim = hidden_dims[-1]
        
        for i, hidden_dim in enumerate(reversed_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(reversed_dims) - 1:  # No activation on output layer
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout_rate))
            else:
                decoder_layers.append(nn.Sigmoid())  # Output activation
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Encode to bottleneck
        hidden = self.encoder(x)
        
        # Decode from bottleneck
        reconstructed = self.decoder(hidden)
        
        return reconstructed, hidden # Necessary for debugging and sanity check
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, hidden):
        return self.decoder(hidden)
    
    # BiC assignment 
    def get_bottleneck_weights(self):
        # Find the bottleneck layer (last linear layer in encoder)
        bottleneck_layer = None
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                bottleneck_layer = module
        
        if bottleneck_layer is not None:
            return bottleneck_layer.weight.data
        else:
            raise ValueError("No bottleneck layer found")
    
    def get_reconstruction_weights(self):
        # Find the first linear layer in decoder
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                return module.weight.data
        
        raise ValueError("No decoder linear layer found")
    
    def get_hidden_activations(self, x):
        with torch.no_grad():
            hidden = self.encode(x)
        return hidden

def create_bicluster_model(model_type, input_dim, hidden_config, dropout_rate=0.2):
    
    # customizing biclustering autoencoder model 
    
    if model_type == 'simple':
        return BiclusteringAutoencoder(input_dim, hidden_config, dropout_rate)
    elif model_type == 'deep':
        return BiclusteringAutoencoderDeep(input_dim, hidden_config, dropout_rate)
    else:
        raise ValueError("model_type must be 'simple' or 'deep'")