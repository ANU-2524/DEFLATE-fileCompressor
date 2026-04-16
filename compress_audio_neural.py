"""
Neural Audio Codec Demo and Compression Script
Compress audio files using the Neural Audio Codec
"""

import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from pathlib import Path
import json
import time
import sys

sys.path.insert(0, str(Path(__file__).parent))

from neural_audio.model import NeuralAudioCodec
from neural_audio.evaluator import AudioEvaluator, CompressionBenchmark
from huffman.encoder import HuffmanEncoder  # Your existing Huffman encoder
import zlib


class NeuralAudioCompressor:
    """
    Audio compression using Neural Codec integrated with Huffman/Entropy coding
    """
    
    def __init__(self, checkpoint_path=None, device='cpu'):
        """
        Initialize compressor.
        
        Args:
            checkpoint_path: Path to pretrained codec checkpoint
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.codec = NeuralAudioCodec(latent_dim=128).to(device)
        self.evaluator = AudioEvaluator()
        
        # Load checkpoint if provided
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.codec.load_state_dict(checkpoint['model_state'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        
        self.codec.eval()
    
    def compress_audio(self, audio_path, output_path=None, use_entropy_coding=True):
        """
        Compress audio file using neural codec.
        
        Args:
            audio_path: Path to input audio file (WAV)
            output_path: Path to save compressed data
            use_entropy_coding: Whether to apply Huffman coding to indices
        
        Returns:
            compression_info: Dict with compression details
        """
        print(f"\nCompressing: {audio_path}")
        
        # Load audio
        audio_data, sr = sf.read(audio_path)
        original_length = len(audio_data)
        
        # Convert to target sample rate if needed
        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        
        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 1e-4:
            audio_data = audio_data / max_val
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
        # Reshape to [1, 1, num_samples]
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        
        original_size = audio_data.nbytes
        
        # Compress using neural codec
        with torch.no_grad():
            indices, latent_shape = self.codec.encode(audio_tensor.to(self.device))
        
        indices_np = indices.cpu().numpy()
        
        # Optional: Apply additional entropy coding
        if use_entropy_coding:
            # Convert indices to bytes and apply Huffman coding
            indices_bytes = indices_np.astype(np.uint8).tobytes()
            compressed_bytes = zlib.compress(indices_bytes, level=9)
        else:
            compressed_bytes = indices_np.astype(np.uint8).tobytes()
        
        compressed_size = len(compressed_bytes)
        
        # Save compressed data
        if output_path:
            compression_data = {
                'indices_shape': indices_np.shape,
                'latent_shape': list(latent_shape),
                'original_length': original_length,
                'sample_rate': 16000,
                'compressed_indices': compressed_bytes.hex(),
            }
            
            with open(output_path, 'w') as f:
                json.dump(compression_data, f)
            print(f"Saved compressed audio to: {output_path}")
        
        # Calculate metrics
        compression_ratio = original_size / (compressed_size + 1e-10)
        duration = original_length / 16000
        bitrate_kbps = (len(compressed_bytes) * 8 / 1000) / duration
        
        info = {
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'compression_ratio': compression_ratio,
            'bitrate_kbps': bitrate_kbps,
            'duration_seconds': duration,
            'indices_shape': indices_np.shape,
        }
        
        print(f"Original size: {original_size:,} bytes")
        print(f"Compressed size: {compressed_size:,} bytes")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Bitrate: {bitrate_kbps:.2f} kbps")
        print(f"Duration: {duration:.2f} seconds")
        
        return info
    
    def decompress_audio(self, compressed_path, output_path=None):
        """
        Decompress audio from compressed format.
        
        Args:
            compressed_path: Path to compressed audio
            output_path: Path to save reconstructed audio
        
        Returns:
            reconstructed_audio: NumPy array of audio
        """
        print(f"\nDecompressing: {compressed_path}")
        
        # Load compressed data
        with open(compressed_path, 'r') as f:
            compression_data = json.load(f)
        
        # Decompress indices
        compressed_bytes = bytes.fromhex(compression_data['compressed_indices'])
        try:
            indices_bytes = zlib.decompress(compressed_bytes)
        except:
            indices_bytes = compressed_bytes
        
        indices_np = np.frombuffer(indices_bytes, dtype=np.uint8)
        indices_np = indices_np.reshape(compression_data['indices_shape'])
        indices = torch.from_numpy(indices_np).to(self.device)
        
        latent_shape = compression_data['latent_shape']
        
        # Decode
        with torch.no_grad():
            audio_reconstructed = self.codec.decode(indices, tuple(latent_shape))
        
        audio_np = audio_reconstructed.cpu().numpy().squeeze()
        
        # Save if requested
        if output_path:
            sf.write(output_path, audio_np, 16000)
            print(f"Saved reconstructed audio to: {output_path}")
        
        return audio_np
    
    def compare_with_traditional(self, audio_path):
        """
        Compare neural codec with traditional compression methods.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            comparison: Dict with comparison results
        """
        print(f"\n{'='*70}")
        print("COMPRESSION METHOD COMPARISON")
        print(f"{'='*70}")
        
        # Load audio
        audio_data, sr = sf.read(audio_path)
        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        
        original_size = audio_data.nbytes
        duration = len(audio_data) / 16000
        
        methods = {}
        
        # 1. Neural Codec
        print("\n1. Neural Audio Codec")
        neural_info = self.compress_audio(audio_path)
        methods['Neural Codec'] = neural_info
        
        # 2. DEFLATE (run-length + gzip)
        print("\n2. DEFLATE (gzip compression)")
        deflate_compressed = zlib.compress(audio_data.tobytes(), level=9)
        deflate_size = len(deflate_compressed)
        deflate_ratio = original_size / deflate_size
        deflate_bitrate = (deflate_size * 8 / 1000) / duration
        methods['DEFLATE'] = {
            'original_size_bytes': original_size,
            'compressed_size_bytes': deflate_size,
            'compression_ratio': deflate_ratio,
            'bitrate_kbps': deflate_bitrate,
        }
        print(f"DEFLATE compressed size: {deflate_size:,} bytes")
        print(f"DEFLATE compression ratio: {deflate_ratio:.2f}x")
        print(f"DEFLATE bitrate: {deflate_bitrate:.2f} kbps")
        
        # 3. LZMA (7zip)
        print("\n3. LZMA (xz compression)")
        import lzma
        lzma_compressed = lzma.compress(audio_data.tobytes(), preset=9)
        lzma_size = len(lzma_compressed)
        lzma_ratio = original_size / lzma_size
        lzma_bitrate = (lzma_size * 8 / 1000) / duration
        methods['LZMA'] = {
            'original_size_bytes': original_size,
            'compressed_size_bytes': lzma_size,
            'compression_ratio': lzma_ratio,
            'bitrate_kbps': lzma_bitrate,
        }
        print(f"LZMA compressed size: {lzma_size:,} bytes")
        print(f"LZMA compression ratio: {lzma_ratio:.2f}x")
        print(f"LZMA bitrate: {lzma_bitrate:.2f} kbps")
        
        # 4. Zstandard
        print("\n4. Zstandard (Zstd)")
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=22)
        zstd_compressed = cctx.compress(audio_data.tobytes())
        zstd_size = len(zstd_compressed)
        zstd_ratio = original_size / zstd_size
        zstd_bitrate = (zstd_size * 8 / 1000) / duration
        methods['Zstandard'] = {
            'original_size_bytes': original_size,
            'compressed_size_bytes': zstd_size,
            'compression_ratio': zstd_ratio,
            'bitrate_kbps': zstd_bitrate,
        }
        print(f"Zstd compressed size: {zstd_size:,} bytes")
        print(f"Zstd compression ratio: {zstd_ratio:.2f}x")
        print(f"Zstd bitrate: {zstd_bitrate:.2f} kbps")
        
        # Print comparison table
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"{'Method':<20} {'Ratio':<12} {'Bitrate (kbps)':<18} {'Size (bytes)':<15}")
        print("-"*70)
        
        for method, info in methods.items():
            ratio = info['compression_ratio']
            bitrate = info['bitrate_kbps']
            size = info['compressed_size_bytes']
            print(f"{method:<20} {ratio:<12.2f}x {bitrate:<18.2f} {size:<15,}")
        
        print(f"{'='*70}")
        
        return methods


def main():
    """Demo script"""
    
    # Check if audio file exists
    audio_file = Path("test.wav")
    if not audio_file.exists():
        print("Creating test audio file...")
        # Generate test audio
        sr = 16000
        duration = 5  # seconds
        t = np.linspace(0, duration, sr * duration)
        audio = np.sin(2 * np.pi * 440 * t) * 0.3 + np.random.randn(len(t)) * 0.05
        sf.write(str(audio_file), audio, sr)
        print(f"Created {audio_file}")
    
    # Initialize compressor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compressor = NeuralAudioCompressor(device=device)
    
    # Compress audio
    compression_info = compressor.compress_audio(
        str(audio_file),
        output_path='compressed_audio.json'
    )
    
    # Decompress
    audio_reconstructed = compressor.decompress_audio(
        'compressed_audio.json',
        output_path='reconstructed_audio.wav'
    )
    
    # Compare with traditional methods
    compressor.compare_with_traditional(str(audio_file))
    
    print("\nDemo complete!")
    print("Files created:")
    print("  - compressed_audio.json: Compressed audio data")
    print("  - reconstructed_audio.wav: Reconstructed audio")


if __name__ == '__main__':
    main()
