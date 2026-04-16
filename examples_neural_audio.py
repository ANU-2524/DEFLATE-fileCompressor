"""
Neural Audio Compression - Quick Start Examples
Shows how to use each component
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from neural_audio.model import NeuralAudioCodec
from neural_audio.encoder import AudioEncoder
from neural_audio.decoder import AudioDecoder
from neural_audio.quantizer import VectorQuantizer
from neural_audio.data_loader import SyntheticAudioDataset, DataLoaderFactory
from neural_audio.trainer import AudioCompressionTrainer
from neural_audio.evaluator import AudioEvaluator


def example_1_basic_usage():
    """
    Example 1: Basic codec usage
    Shows how to encode and decode audio
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    # Create codec
    codec = NeuralAudioCodec(latent_dim=128, num_codebook_entries=256)
    print(f"Created codec: {codec}")
    
    # Create dummy audio
    batch_size = 2
    audio = torch.randn(batch_size, 1, 16000)  # 2 samples, 1 second each
    print(f"\nInput audio shape: {audio.shape}")
    print(f"Input audio range: [{audio.min():.3f}, {audio.max():.3f}]")
    
    # Forward pass (encode + decode)
    reconstructed, losses = codec(audio)
    print(f"\nReconstructed shape: {reconstructed.shape}")
    print(f"Reconstruction loss: {losses['reconstruction_loss']:.6f}")
    print(f"VQ loss: {losses['vq_loss']:.6f}")
    
    # Compression metrics
    compression_ratio = codec.get_compression_ratio()
    if compression_ratio:
        print(f"\nCompression ratio: {compression_ratio:.2f}x")


def example_2_encoder_decoder():
    """
    Example 2: Using encoder and decoder separately
    Shows latent space compression
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Encoder-Decoder Breakdown")
    print("="*70)
    
    # Create components
    encoder = AudioEncoder(latent_dim=128)
    decoder = AudioDecoder(latent_dim=128)
    
    # Audio
    audio = torch.randn(1, 1, 16000)
    print(f"Original audio shape: {audio.shape}  → {audio.numel()} values")
    
    # Encode
    latent = encoder(audio)
    print(f"Latent space shape: {latent.shape}  → {latent.numel()} values")
    print(f"Compression in latent: {audio.numel() / latent.numel():.1f}x")
    
    # Decode
    reconstructed = decoder(latent)
    print(f"Reconstructed audio shape: {reconstructed.shape}")
    
    # Error
    error = torch.mean((audio - reconstructed) ** 2)
    print(f"\nReconstruction MSE: {error:.6f}")


def example_3_quantization():
    """
    Example 3: Quantization layer
    Shows how continuous values become discrete
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Quantization")
    print("="*70)
    
    quantizer = VectorQuantizer(num_embeddings=256, embedding_dim=128)
    
    # Continuous latent values
    latent = torch.randn(2, 128, 100)  # 2 samples, 128 dimensions, 100 timesteps
    print(f"Continuous latent: {latent.shape}")
    print(f"  Values: {latent[0, 0, :5]}")
    
    # Quantize
    latent_q, vq_loss, indices = quantizer(latent)
    print(f"\nAfter quantization:")
    print(f"  Quantized shape: {latent_q.shape}")
    print(f"  Indices shape: {indices.shape}")
    print(f"  Indices (9-bit): {indices[0, :5]}")
    print(f"  VQ Loss: {vq_loss:.6f}")
    
    # Compression info
    bits_per_index = np.log2(256)
    total_indices = indices.numel()
    compressed_bits = total_indices * bits_per_index
    original_bits = 32 * latent.numel()  # Float32
    ratio = original_bits / compressed_bits
    print(f"\nCompression via quantization: {ratio:.1f}x")
    print(f"  Original: {original_bits/1024:.1f} KB (float32)")
    print(f"  Quantized: {compressed_bits/1024:.1f} KB (8-bit indices)")


def example_4_data_loading():
    """
    Example 4: Loading audio data
    Shows dataset and dataloader usage
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Data Loading")
    print("="*70)
    
    # Create synthetic dataset
    dataset = SyntheticAudioDataset(num_samples=100, sample_rate=16000, chunk_length=16000)
    print(f"Created dataset with {len(dataset)} samples")
    
    # Create dataloader
    loader = DataLoaderFactory.create_synthetic_loader(
        num_samples=100,
        batch_size=4,
        sample_rate=16000
    )
    print(f"Created dataloader with batch_size=4")
    
    # Iterate
    for batch_idx, audio_batch in enumerate(loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Shape: {audio_batch.shape}")
        print(f"  Range: [{audio_batch.min():.3f}, {audio_batch.max():.3f}]")
        print(f"  Mean: {audio_batch.mean():.3f}")
        print(f"  Std: {audio_batch.std():.3f}")
        if batch_idx >= 2:
            break


def example_5_evaluation():
    """
    Example 5: Evaluation metrics
    Shows how to evaluate codec quality
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Evaluation Metrics")
    print("="*70)
    
    evaluator = AudioEvaluator(sample_rate=16000)
    
    # Create test data
    original = torch.randn(1, 1, 16000)
    
    # Slightly degraded reconstruction (simulating compression loss)
    noise = torch.randn_like(original) * 0.05
    reconstructed = original + noise
    
    print(f"Original audio: {original.shape}")
    print(f"Reconstructed audio: {reconstructed.shape}")
    
    # Evaluate
    metrics = evaluator.evaluate_reconstruction(original, reconstructed)
    
    print(f"\nReconstruction Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Spectral distance
    spec_dist = evaluator.spectral_distance(original, reconstructed)
    if spec_dist:
        print(f"  Spectral distance: {spec_dist:.6f}")


def example_6_training_mini():
    """
    Example 6: Mini training loop
    Shows how to train the codec
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Training (Mini - 2 Epochs)")
    print("="*70)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    codec = NeuralAudioCodec(latent_dim=128).to(device)
    optimizer = torch.optim.Adam(codec.parameters(), lr=1e-3)
    
    # Mini dataset
    loader = DataLoaderFactory.create_synthetic_loader(
        num_samples=20,
        batch_size=4,
        sample_rate=16000
    )
    
    # Train
    num_epochs = 2
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, audio in enumerate(loader):
            audio = audio.to(device)
            
            # Forward
            optimizer.zero_grad()
            reconstructed, losses = codec(audio)
            
            # Loss
            vq_loss = losses['vq_loss']
            recon_loss = losses['reconstruction_loss']
            total_loss = 0.25 * vq_loss + recon_loss
            
            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
    
    print("\nTraining complete!")


def example_7_compression_ratio():
    """
    Example 7: Understanding compression ratios
    Shows relationship between components
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Compression Breakdown")
    print("="*70)
    
    # Original audio
    sample_rate = 16000
    duration = 5  # seconds
    num_samples = sample_rate * duration
    
    print(f"Original audio:")
    print(f"  Duration: {duration}s @ {sample_rate}Hz")
    print(f"  Samples: {num_samples}")
    print(f"  As float32: {num_samples * 4 / 1024:.1f} KB")
    print(f"  As int16 (normal): {num_samples * 2 / 1024:.1f} KB")
    
    # After neural compression
    print(f"\nAfter Neural Codec compression:")
    codec = NeuralAudioCodec(latent_dim=128, num_codebook_entries=256)
    
    # Latent space
    latent_dim = 128
    temporal_compression = 16  # 16x from encoder
    latent_length = num_samples // temporal_compression
    
    print(f"  Latent dim: {latent_dim}")
    print(f"  Temporal compression: {temporal_compression}x")
    print(f"  Latent length: {latent_length}")
    print(f"  Latent values: {latent_dim * latent_length}")
    
    # After quantization
    bits_per_index = np.log2(256)
    total_bits = latent_dim * latent_length * bits_per_index
    total_bytes = total_bits / 8
    
    print(f"\nAfter Quantization (8-bit):")
    print(f"  Total indices: {latent_dim * latent_length}")
    print(f"  Bits per index: {bits_per_index:.1f}")
    print(f"  Total bits: {total_bits:.0f}")
    print(f"  Size: {total_bytes / 1024:.1f} KB")
    
    # Bitrate
    bitrate_kbps = (total_bytes * 8 / 1000) / duration
    print(f"  Bitrate: {bitrate_kbps:.1f} kbps")
    
    # Compression ratio
    original_bits = num_samples * 16  # 16-bit audio
    ratio = original_bits / total_bits
    print(f"\nOverall compression ratio: {ratio:.1f}x")
    print(f"Compared to MP3 (32 kbps): {256/bitrate_kbps:.1f}x better")
    print(f"Compared to Opus (20 kbps): {160/bitrate_kbps:.1f}x better")


def main():
    """Run all examples"""
    print("\n")
    print("█" * 70)
    print("  NEURAL AUDIO COMPRESSION - EXAMPLES")
    print("█" * 70)
    
    example_1_basic_usage()
    example_2_encoder_decoder()
    example_3_quantization()
    example_4_data_loading()
    example_5_evaluation()
    example_6_training_mini()
    example_7_compression_ratio()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nNext steps:")
    print("1. Read NEURAL_AUDIO_README.md for detailed documentation")
    print("2. Run: python train_neural_codec.py --epochs 50 --use-synthetic")
    print("3. Run: python compress_audio_neural.py to see compression demo")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
