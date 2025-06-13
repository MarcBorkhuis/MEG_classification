"""Neural Network Models for MEG Classification

This module implements various neural network architectures for MEG classification:
- EEGNet: Lightweight CNN architecture optimized for EEG/MEG data
- MEEGNet: Modified CNN architecture for MEG data
- MEGNetMultiHead: Transformer-based architecture with multi-head attention

Each architecture is designed to handle the unique characteristics of MEG data,
including temporal dynamics and spatial relationships between sensors.
"""

import torch
from torch import nn
import numpy as np
import math

# ---------------------------------------------------------------------------- #
#                           Custom Network Layers                              #
# ---------------------------------------------------------------------------- #

class Flatten(nn.Module):
    """Flattens the input tensor while preserving the batch dimension.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, ...)
        
    Returns:
        torch.Tensor: Flattened tensor of shape (batch_size, -1)
    """
    def forward(self, x):
        return x.view(x.size(0), -1)

class DepthwiseConv2d(nn.Conv2d):
    """Depthwise separable convolution layer.
    
    This layer performs a depthwise convolution where each input channel
    is convolved with its own set of filters.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution
        padding (int or tuple, optional): Zero-padding added to both sides
        dilation (int or tuple, optional): Spacing between kernel elements
        bias (bool, optional): If True, adds a learnable bias
        padding_mode (str, optional): Type of padding to use
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, bias=False, padding_mode='zeros'):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups=in_channels, bias=bias, padding_mode=padding_mode
        )

class SeparableConv2d(nn.Module):
    """Separable convolution module combining depthwise and pointwise convolutions.
    
    This module first applies a depthwise convolution to each input channel,
    followed by a pointwise convolution that combines the features.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the convolving kernel
        bias (bool, optional): If True, adds a learnable bias
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            groups=in_channels, bias=bias, padding='same'
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# ---------------------------------------------------------------------------- #
#                          Positional Encoding                                 #
# ---------------------------------------------------------------------------- #

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer architectures.
    
    This module adds positional information to the input sequence using
    sine and cosine functions of different frequencies.
    
    Args:
        d_model (int): Dimension of the model
        dropout (float, optional): Dropout probability
        max_len (int, optional): Maximum sequence length
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# ---------------------------------------------------------------------------- #
#                            Base Network Class                                #
# ---------------------------------------------------------------------------- #

class CustomNet(nn.Module):
    """Base class for all network architectures.
    
    This class provides a common interface for all network architectures,
    with separate feature extraction and classification components.
    
    Args:
        input_size (tuple): Shape of input data (excluding batch dimension)
        n_outputs (int): Number of output classes
    """
    def __init__(self, input_size: tuple, n_outputs: int):
        super().__init__()
        self.input_size = input_size
        self.n_outputs = n_outputs
        self.feature_extraction = nn.Sequential()
        self.classif = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Network output
        """
        features = self.get_features(x)
        return self.classif(features).squeeze()
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Extracted features
        """
        return self.feature_extraction(x)

    def get_lin_size(self) -> int:
        """Calculate the size of the linear layer input.
        
        Returns:
            int: Number of features after feature extraction
        """
        with torch.no_grad():
            dummy_input = torch.zeros((1, *self.input_size))
            output_shape = self.feature_extraction(dummy_input).shape
        return np.prod(output_shape[1:])

# ---------------------------------------------------------------------------- #
#                           Network Architectures                              #
# ---------------------------------------------------------------------------- #

class EEGNet(CustomNet):
    """EEGNet architecture optimized for EEG/MEG data.
    
    This architecture uses depthwise separable convolutions and temporal
    convolutions to efficiently process EEG/MEG signals.
    
    Args:
        input_size (tuple): Shape of input data
        n_outputs (int): Number of output classes
        hparams (dict): Hyperparameters including:
            - n_filters_1 (int): Number of temporal filters
            - depth_multiplier (int): Depth multiplier for spatial filters
            - n_filters_2 (int): Number of pointwise filters
            - kernel_length (int): Length of temporal convolution kernel
            - dropout (float): Dropout probability
    """
    def __init__(self, input_size, n_outputs, hparams: dict):
        super().__init__(input_size, n_outputs)
        F1 = hparams.get('n_filters_1', 8)
        D = hparams.get('depth_multiplier', 2)
        F2 = hparams.get('n_filters_2', F1 * D)
        kernel_length = hparams.get('kernel_length', 64)
        dropout = hparams.get('dropout', 0.25)
        n_channels = input_size[1]

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            DepthwiseConv2d(F1, F1 * D, (n_channels, 1), bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            SeparableConv2d(F1 * D, F2, (1, 16)),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            Flatten()
        )
        self.classif = nn.Linear(self.get_lin_size(), n_outputs)

class MEEGNet(CustomNet):
    """Modified CNN architecture for MEG data.
    
    This architecture uses spatial and temporal convolutions to process
    MEG data, with a focus on capturing both spatial and temporal patterns.
    
    Args:
        input_size (tuple): Shape of input data
        n_outputs (int): Number of output classes
        hparams (dict): Hyperparameters including:
            - n_filters_1 (int): Number of spatial filters
            - n_filters_2 (int): Number of temporal filters
            - dropout (float): Dropout probability
    """
    def __init__(self, input_size, n_outputs, hparams: dict):
        super().__init__(input_size, n_outputs)
        n_filters_1 = hparams.get('n_filters_1', 16)
        n_filters_2 = hparams.get('n_filters_2', 32)
        dropout = hparams.get('dropout', 0.5)
        n_channels = input_size[1]
        
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, n_filters_1, (n_channels, 1), padding='valid'),
            nn.BatchNorm2d(n_filters_1),
            nn.ELU(),
            nn.Conv2d(n_filters_1, n_filters_2, (1, 16), padding='same'),
            nn.BatchNorm2d(n_filters_2),
            nn.ELU(),
            nn.MaxPool2d((1, 4)),
            nn.Dropout(dropout),
            Flatten()
        )
        self.classif = nn.Linear(self.get_lin_size(), n_outputs)

class MEGNetMultiHead(CustomNet):
    """Transformer-based model for MEG classification.
    
    This architecture uses a combination of convolutional layers and
    transformer encoder layers to process MEG data, with multi-head
    attention for capturing complex temporal dependencies.
    
    Args:
        input_size (tuple): Shape of input data
        n_outputs (int): Number of output classes
        hparams (dict): Hyperparameters including:
            - n_filters_1 (int): Number of initial filters
            - n_head (int): Number of attention heads
            - n_layers (int): Number of transformer layers
            - dropout (float): Dropout probability
            - d_model (int): Dimension of the transformer model
    """
    def __init__(self, input_size, n_outputs, hparams: dict):
        super().__init__(input_size, n_outputs)
        n_channels = input_size[1]
        
        n_filters = hparams.get('n_filters_1', 32)
        n_head = hparams.get('n_head', 4)
        n_layers = hparams.get('n_layers', 2)
        dropout = hparams.get('dropout', 0.2)
        d_model = hparams.get('d_model', 128)

        self.conv_extractor = nn.Sequential(
            nn.Conv2d(1, n_filters, (n_channels, 1), padding='valid'),
            nn.BatchNorm2d(n_filters),
            nn.Conv2d(n_filters, d_model, (1, 25), padding='same'),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.MaxPool2d((1, 4))
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros((1, *input_size))
            conv_out_shape = self.conv_extractor(dummy_input).shape
            seq_len = conv_out_shape[3]
        
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_model * 2, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.feature_extraction = nn.Identity()
        self.classif = nn.Linear(d_model * seq_len, n_outputs)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using convolutional and transformer layers.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Extracted features
        """
        x = self.conv_extractor(x)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, Seq, Feat)
        x = self.pos_encoder(x.permute(1,0,2)).permute(1,0,2)
        x = self.transformer_encoder(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Network output
        """
        features = self.get_features(x)
        return self.classif(features)

# ---------------------------------------------------------------------------- #
#                             Network Factory                                  #
# ---------------------------------------------------------------------------- #

def create_net(net_option: str, input_size: tuple, n_outputs: int, hparams: dict) -> nn.Module:
    """Create a neural network based on the specified option.
    
    Args:
        net_option (str): Name of the network architecture
        input_size (tuple): Shape of input data
        n_outputs (int): Number of output classes
        hparams (dict): Hyperparameters for the network
        
    Returns:
        nn.Module: The created network
        
    Raises:
        AttributeError: If the network option is invalid
    """
    net_options = {
        "eegnet": EEGNet,
        "meegnet": MEEGNet,
        "attention": MEGNetMultiHead,
    }
    net_option = net_option.lower()
    if net_option not in net_options:
        raise AttributeError(f"Invalid network option: {net_option}")
    return net_options[net_option](input_size, n_outputs, hparams)
