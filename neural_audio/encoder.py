"""
Neural Audio Encoder
Compresses audio to latent space representation
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with convolution"""
    
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual  # Skip connection
        return self.relu(out)


class AudioEncoder(nn.Module):
    """
    Encodes audio to compressed latent representation.
    
    Input: [batch, 1, num_samples]
    Output: [batch, latent_dim, compressed_length]
    
    Example:
        encoder = AudioEncoder(latent_dim=128)
        audio = torch.randn(4, 1, 16000)  # 4 samples, 1 channel, 16k samples
        latent = encoder(audio)  # [4, 128, 250]  -- audio compressed 64x
    """
    
    def __init__(self, latent_dim=128, num_channels=1):
        """
        Args:
            latent_dim: Dimension of latent representation (bottleneck)
            num_channels: Audio channels (1 for mono, 2 for stereo)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        
        # Input projection
        self.input_conv = nn.Conv1d(num_channels, 64, kernel_size=7, stride=1, padding=3)
        
        # Downsampling blocks with residual connections
        # Stage 1: Stride 2 (2x compression)
        self.down1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        
        # Stage 2: Stride 2 (4x compression total)
        self.down2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(256),
            ResidualBlock(256),
        )
        
        # Stage 3: Stride 4 (16x compression total)
        self.down3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            ResidualBlock(512),
            ResidualBlock(512),
        )
        
        # Bottleneck - compress to latent dimension
        self.bottleneck = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, latent_dim, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, num_channels, num_samples]
        
        Returns:
            latent: [batch, latent_dim, compressed_length]
        """
        # Input projection
        x = self.input_conv(x)
        
        # Downsampling stages
        x = self.down1(x)  # 2x compression
        x = self.down2(x)  # 4x compression
        x = self.down3(x)  # 16x compression
        
        # Bottleneck
        latent = self.bottleneck(x)
        
        return latent


class ConditionalEncoder(nn.Module):
    """
    Encoder with conditional features support
    (for future multi-codec functionality)
    """
    
    def __init__(self, latent_dim=128, num_channels=1, embedding_dim=16):
        super().__init__()
        self.encoder = AudioEncoder(latent_dim, num_channels)
        self.embedding_dim = embedding_dim
    
    def forward(self, x, condition=None):
        """
        Args:
            x: [batch, num_channels, num_samples]
            condition: Optional conditioning information
        
        Returns:
            latent: [batch, latent_dim, compressed_length]
        """
        latent = self.encoder(x)
        return latent
