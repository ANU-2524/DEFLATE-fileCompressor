"""
Vector Quantization Layer
Converts continuous latent codes to discrete integers for compression
"""

import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    """
    Vector Quantization - Maps continuous vectors to nearest codebook entries.
    
    This is a "straight-through estimator" - during forward pass, it quantizes
    the input, and during backward pass, gradients pass through as if no 
    quantization happened (allowing training).
    
    Example:
        vq = VectorQuantizer(num_embeddings=256, embedding_dim=128)
        latent = torch.randn(4, 128, 250)
        quantized, indices = vq(latent)
        # quantized: [4, 128, 250] - quantized version
        # indices: [4, 250] - which codebook entry was used (for compression)
    """
    
    def __init__(self, num_embeddings=256, embedding_dim=128, beta=0.25):
        """
        Args:
            num_embeddings: Size of codebook (number of discrete values)
            embedding_dim: Dimension of each embedding (latent_dim)
            beta: Commitment loss weight
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        
        # Codebook - learnable embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize with normal distribution
        self.embedding.weight.data.uniform_(-1./num_embeddings, 1./num_embeddings)
    
    def forward(self, z):
        """
        Args:
            z: [batch, embedding_dim, time_steps]
        
        Returns:
            z_q: [batch, embedding_dim, time_steps] - quantized version
            loss: Scalar - quantization loss (for training)
            indices: [batch, time_steps] - which codebook entry was selected
        """
        # Flatten spatial dimensions
        # [batch, embedding_dim, time] -> [batch*time, embedding_dim]
        z_flat = z.permute(0, 2, 1).reshape(-1, self.embedding_dim)
        
        # Calculate distances to all codebook entries
        # Using simplified calculation: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight ** 2, dim=1) -
            2 * torch.matmul(z_flat, self.embedding.weight.t())
        )
        # [batch*time, num_embeddings]
        
        # Get nearest codebook indices
        indices = torch.argmin(distances, dim=1)
        # [batch*time]
        
        # Get quantized values
        z_q_flat = self.embedding(indices)
        # [batch*time, embedding_dim]
        
        # Reshape back
        z_q = z_q_flat.reshape(z.shape[0], z.shape[2], self.embedding_dim)
        z_q = z_q.permute(0, 2, 1)
        # [batch, embedding_dim, time]
        
        # Calculate loss (for training)
        # VQ Loss = ||sg[z] - e||^2 + beta * ||z - sg[e]||^2
        # sg = stop-gradient (detach)
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        
        # Straight-through estimator: copy gradients from quantized to input
        z_q = z + (z_q - z).detach()
        
        # Reshape indices back
        indices = indices.reshape(z.shape[0], z.shape[2])
        # [batch, time_steps]
        
        return z_q, loss, indices
    
    def get_codebook(self):
        """Return the codebook embeddings"""
        return self.embedding.weight.data


class ScalarQuantizer(nn.Module):
    """
    Simple scalar quantization - quantizes each value independently.
    More memory efficient but less powerful than vector quantization.
    """
    
    def __init__(self, levels=256, value_range=2.0):
        """
        Args:
            levels: Number of quantization levels
            value_range: Expected range of input values [-value_range, value_range]
        """
        super().__init__()
        self.levels = levels
        self.value_range = value_range
        self.step = (2 * value_range) / (levels - 1)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of any shape
        
        Returns:
            x_q: Quantized version
            loss: Quantization error
            indices: Which level was selected
        """
        # Clamp to expected range
        x_clamped = torch.clamp(x, -self.value_range, self.value_range)
        
        # Convert to indices
        indices = torch.round((x_clamped + self.value_range) / self.step).long()
        indices = torch.clamp(indices, 0, self.levels - 1)
        
        # Quantize back
        x_q = (indices.float() * self.step) - self.value_range
        
        # Loss = quantization error
        loss = torch.mean((x_q.detach() - x) ** 2)
        
        # Straight-through estimator
        x_q = x + (x_q - x).detach()
        
        return x_q, loss, indices
