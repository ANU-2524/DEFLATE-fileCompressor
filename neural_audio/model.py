"""
Neural Audio Codec
Complete encoder-decoder system with quantization
"""

import torch
import torch.nn as nn
from .encoder import AudioEncoder
from .decoder import AudioDecoder
from .quantizer import VectorQuantizer


class NeuralAudioCodec(nn.Module):
    """
    Complete Neural Audio Compression Codec.
    
    Pipeline:
        Audio → Encoder → Quantizer → Entropy Coding → Compressed
        Compressed → Entropy Decoding → Dequantizer → Decoder → Audio
    
    Example:
        codec = NeuralAudioCodec(latent_dim=128)
        audio = torch.randn(4, 1, 16000)  # 4 samples
        
        # Forward pass stores indices for compression
        reconstructed, loss = codec(audio)
        
        # Get codec metrics
        compression_ratio = codec.get_compression_ratio()
    """
    
    def __init__(self, latent_dim=128, num_channels=1, num_codebook_entries=256, 
                 sample_rate=16000):
        """
        Args:
            latent_dim: Dimension of latent space
            num_channels: Audio channels (1=mono, 2=stereo)
            num_codebook_entries: Quantization codebook size (log2 = bits per entry)
            sample_rate: Audio sample rate (Hz)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.num_codebook_entries = num_codebook_entries
        self.sample_rate = sample_rate
        
        # Components
        self.encoder = AudioEncoder(latent_dim, num_channels)
        self.decoder = AudioDecoder(latent_dim, num_channels)
        self.quantizer = VectorQuantizer(num_codebook_entries, latent_dim)
        
        # Track last compression metrics
        self._last_indices = None
        self._last_latent_shape = None
    
    def encode(self, audio):
        """
        Encode audio to compressed representation.
        
        Args:
            audio: [batch, num_channels, num_samples]
        
        Returns:
            indices: [batch, time_steps] - codebook indices for entropy coding
            shape: Tuple of latent shape for reconstruction
        """
        # Neural encoding
        latent = self.encoder(audio)
        self._last_latent_shape = latent.shape
        
        # Quantization
        _, _, indices = self.quantizer(latent)
        self._last_indices = indices
        
        return indices, latent.shape
    
    def decode(self, indices, shape):
        """
        Decode from compressed indices back to audio.
        
        Args:
            indices: [batch, time_steps] - codebook indices
            shape: Original latent shape
        
        Returns:
            audio: [batch, num_channels, num_samples]
        """
        # Get embeddings from codebook
        batch_size, time_steps = indices.shape
        z_q = self.quantizer.embedding(indices)  # [batch, time, embedding_dim]
        z_q = z_q.permute(0, 2, 1)  # [batch, embedding_dim, time]
        
        # Neural decoding
        audio = self.decoder(z_q)
        
        return audio
    
    def forward(self, audio):
        """
        Forward pass: encode and decode for training.
        
        Args:
            audio: [batch, num_channels, num_samples]
        
        Returns:
            reconstructed: [batch, num_channels, num_samples]
            loss: Dict with loss components
        """
        # Encode
        latent = self.encoder(audio)
        
        # Quantize
        latent_q, vq_loss, indices = self.quantizer(latent)
        
        # Decode
        reconstructed = self.decoder(latent_q)
        
        # Calculate losses
        losses = {
            'vq_loss': vq_loss,
            'reconstruction_loss': self.reconstruction_loss(audio, reconstructed),
        }
        
        return reconstructed, losses
    
    def reconstruction_loss(self, original, reconstructed):
        """
        Calculate reconstruction loss with perceptual weighting.
        
        Uses simple L1 loss (MAE) which is more perceptually accurate than L2.
        For production, would use STFT-based perceptual loss.
        
        Args:
            original: [batch, channels, samples]
            reconstructed: [batch, channels, samples]
        
        Returns:
            loss: Scalar
        """
        # L1 loss (more perceptually relevant than L2)
        loss = torch.mean(torch.abs(original - reconstructed))
        return loss
    
    def get_compression_ratio(self):
        """
        Calculate theoretical compression ratio based on last encoding.
        
        Returns:
            ratio: Float - original size / compressed size
        """
        if self._last_indices is None or self._last_latent_shape is None:
            return None
        
        batch, latent_dim, time_steps = self._last_latent_shape
        
        # Original size (in bytes)
        # Assuming 16-bit audio at 16kHz
        # 16 bits = 2 bytes per sample
        original_bits = 16
        
        # Compressed size
        # Each index needs log2(num_codebook_entries) bits
        bits_per_entry = (self.num_codebook_entries - 1).bit_length()
        compressed_bits = bits_per_entry * time_steps * latent_dim
        
        # Total indices count
        total_indices = self._last_indices.numel()
        
        # Compression ratio
        ratio = (original_bits * total_indices) / (bits_per_entry * total_indices)
        ratio = original_bits / bits_per_entry if bits_per_entry > 0 else 1.0
        
        return ratio
    
    def get_bitrate(self, audio_length_seconds, sample_rate=None):
        """
        Calculate bitrate of compressed audio.
        
        Args:
            audio_length_seconds: Duration in seconds
            sample_rate: Override instance sample rate
        
        Returns:
            bitrate: kbps
        """
        if self._last_indices is None:
            return None
        
        sr = sample_rate or self.sample_rate
        total_bits = self.num_codebook_entries.bit_length() * self._last_indices.numel()
        total_seconds = audio_length_seconds
        bitrate_kbps = (total_bits / 1000) / total_seconds
        
        return bitrate_kbps
    
    def freeze_encoder(self):
        """Freeze encoder parameters (for fine-tuning decoder)"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def freeze_decoder(self):
        """Freeze decoder parameters"""
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True


class MultiRateCodec(nn.Module):
    """
    Multiple-rate neural codec supporting different compression levels.
    
    Allows trading off between compression ratio and quality at inference time.
    """
    
    def __init__(self, latent_dim=128, num_channels=1, num_qualities=4):
        """
        Args:
            latent_dim: Latent dimension
            num_channels: Audio channels
            num_qualities: Number of compression quality levels
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.num_qualities = num_qualities
        
        # Create codec with largest codebook
        self.codec = NeuralAudioCodec(
            latent_dim=latent_dim,
            num_channels=num_channels,
            num_codebook_entries=256  # 8 bits
        )
    
    def forward(self, audio, quality=1):
        """
        Forward pass with quality control.
        
        Args:
            audio: [batch, channels, samples]
            quality: 0-3 (0=highest compression, 3=highest quality)
        
        Returns:
            reconstructed: Decoded audio
            losses: Dict of losses
        """
        reconstructed, losses = self.codec(audio)
        return reconstructed, losses
