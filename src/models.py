import torch
from torch import nn
import numpy as np
import math

# ---------------------------------------------------------------------------- #
#                           Custom Network Layers                              #
# ---------------------------------------------------------------------------- #

class Flatten(nn.Module):
    """Flattens the input tensor."""
    def forward(self, x):
        return x.view(x.size(0), -1)

class DepthwiseConv2d(nn.Conv2d):
    """Depthwise separable convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, bias=False, padding_mode='zeros'):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups=in_channels, bias=bias, padding_mode=padding_mode
        )

class SeparableConv2d(nn.Module):
    """Separable convolution module."""
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
    """Injects positional information into the sequence."""
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
    """Base class for all network architectures."""
    def __init__(self, input_size: tuple, n_outputs: int):
        super().__init__()
        self.input_size = input_size
        self.n_outputs = n_outputs
        self.feature_extraction = nn.Sequential()
        self.classif = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x)
        return self.classif(features).squeeze()
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extraction(x)

    def get_lin_size(self) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros((1, *self.input_size))
            output_shape = self.feature_extraction(dummy_input).shape
        return np.prod(output_shape[1:])

# ---------------------------------------------------------------------------- #
#                           Network Architectures                              #
# ---------------------------------------------------------------------------- #

class EEGNet(CustomNet):
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
    """Transformer-based model for MEG classification."""
    def __init__(self, input_size, n_outputs, hparams: dict):
        super().__init__(input_size, n_outputs)
        n_channels = input_size[1]
        
        # Hyperparameters from Optuna trial
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
        
        self.feature_extraction = nn.Identity() # Handled in forward
        self.classif = nn.Linear(d_model * seq_len, n_outputs)

    def forward(self, x):
        x = self.conv_extractor(x)
        x = x.squeeze(2).permute(0, 2, 1) # (B, Seq, Feat)
        x = self.pos_encoder(x.permute(1,0,2)).permute(1,0,2)
        x = self.transformer_encoder(x)
        x = x.flatten(start_dim=1)
        return self.classif(x)

# ---------------------------------------------------------------------------- #
#                             Network Factory                                  #
# ---------------------------------------------------------------------------- #
def create_net(net_option: str, input_size: tuple, n_outputs: int, hparams: dict) -> nn.Module:
    """Creates a neural network based on the specified option."""
    net_options = {
        "eegnet": EEGNet,
        "meegnet": MEEGNet,
        "attention": MEGNetMultiHead,
    }
    net_option = net_option.lower()
    if net_option not in net_options:
        raise AttributeError(f"Invalid network option: {net_option}")
    return net_options[net_option](input_size, n_outputs, hparams)
