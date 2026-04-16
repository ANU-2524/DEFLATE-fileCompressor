"""
Neural Audio Decoder
Reconstructs audio from latent representation
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
        out = out + residual
        return self.relu(out)


class AudioDecoder(nn.Module):
    """
    Decodes compressed latent representation back to audio.
    
    Input: [batch, latent_dim, compressed_length]
    Output: [batch, 1, num_samples]
    
    Example:
        decoder = AudioDecoder(latent_dim=128)
        latent = torch.randn(4, 128, 250)  # Compressed audio
        audio = decoder(latent)  # [4, 1, 16000]  -- reconstructed audio
    """
    
    def __init__(self, latent_dim=128, num_channels=1):
        """
        Args:
            latent_dim: Dimension of latent representation
            num_channels: Audio channels (1 for mono, 2 for stereo)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        
        # Expand latent representation
        self.expand = nn.Sequential(
            nn.Conv1d(latent_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
        )
        
        # Upsampling Stage 1: 4x expansion
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            ResidualBlock(256),
            ResidualBlock(256),
        )
        
        # Upsampling Stage 2: 2x expansion
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        
        # Upsampling Stage 3: 2x expansion
        self.up3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
        )
        
        # Output projection to audio
        self.output_conv = nn.Conv1d(64, num_channels, kernel_size=7, padding=3)
    
    def forward(self, x):
        """
        Args:
            x: [batch, latent_dim, compressed_length]
        
        Returns:
            audio: [batch, num_channels, num_samples]
        """
        # Expand latent
        x = self.expand(x)
        
        # Upsampling stages (reverse of encoding)
        x = self.up1(x)   # 4x expansion
        x = self.up2(x)   # 2x expansion
        x = self.up3(x)   # 2x expansion
        
        # Output projection
        audio = self.output_conv(x)
        
        # Tanh to keep audio in [-1, 1] range
        audio = torch.tanh(audio)
        
        return audio


class ConditionalDecoder(nn.Module):
    """
    Decoder with conditional features support
    """
    
    def __init__(self, latent_dim=128, num_channels=1, embedding_dim=16):
        super().__init__()
        self.decoder = AudioDecoder(latent_dim, num_channels)
        self.embedding_dim = embedding_dim
    
    def forward(self, x, condition=None):
        """
        Args:
            x: [batch, latent_dim, compressed_length]
            condition: Optional conditioning information
        
        Returns:
            audio: [batch, num_channels, num_samples]
        """
        audio = self.decoder(x)
        return audio
