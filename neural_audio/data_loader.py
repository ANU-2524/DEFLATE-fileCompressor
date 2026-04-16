"""
Audio Data Loading and Preprocessing
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class AudioDataset(Dataset):
    """
    PyTorch Dataset for audio files.
    
    Handles loading, resampling, normalization and chunking of audio.
    
    Example:
        dataset = AudioDataset('path/to/audio/files', sample_rate=16000, chunk_length=16000)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
    """
    
    def __init__(self, audio_dir, sample_rate=16000, chunk_length=16000, 
                 num_chunks_per_file=1, normalize=True):
        """
        Args:
            audio_dir: Directory containing .wav files
            sample_rate: Target sample rate (will resample if needed)
            chunk_length: Length of audio chunks (samples)
            num_chunks_per_file: How many random chunks to extract per file
            normalize: Whether to normalize audio to [-1, 1]
        """
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.chunk_length = chunk_length
        self.num_chunks_per_file = num_chunks_per_file
        self.normalize = normalize
        
        # Find all audio files
        self.audio_files = list(self.audio_dir.glob('*.wav')) + \
                          list(self.audio_dir.glob('*.mp3')) + \
                          list(self.audio_dir.glob('*.flac'))
        
        if len(self.audio_files) == 0:
            print(f"Warning: No audio files found in {audio_dir}")
        
        # Create list of (file, chunk_index) pairs
        self.samples = []
        for file in self.audio_files:
            for chunk_idx in range(num_chunks_per_file):
                self.samples.append((file, chunk_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Return audio chunk as normalized tensor"""
        audio_file, chunk_idx = self.samples[idx]
        
        try:
            # Load audio
            audio = self._load_audio(audio_file)
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            # Return silence on error
            audio = np.zeros(self.chunk_length, dtype=np.float32)
        
        # Ensure correct length
        if len(audio) < self.chunk_length:
            # Pad with zeros
            audio = np.pad(audio, (0, self.chunk_length - len(audio)), mode='constant')
        
        # Random chunk extraction
        if len(audio) > self.chunk_length:
            start = np.random.randint(0, len(audio) - self.chunk_length)
            audio = audio[start:start + self.chunk_length]
        
        # Normalize
        if self.normalize:
            # Prevent division by zero
            max_val = np.max(np.abs(audio))
            if max_val > 1e-4:
                audio = audio / max_val
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        # Add channel dimension: [length] -> [1, length]
        audio_tensor = audio_tensor.unsqueeze(0)
        
        return audio_tensor
    
    def _load_audio(self, file_path):
        """Load audio file and resample to target sample rate"""
        try:
            import librosa
            audio, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=True)
            return audio
        except ImportError:
            # Fallback to scipy if librosa not available
            from scipy.io import wavfile
            sr, audio = wavfile.read(file_path)
            # Convert to float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32) / (2**15)  # Assuming 16-bit
            # Resample if needed
            if sr != self.sample_rate:
                from scipy import signal
                num_samples = int(len(audio) * self.sample_rate / sr)
                audio = signal.resample(audio, num_samples)
            return audio


class SyntheticAudioDataset(Dataset):
    """
    Generate synthetic audio for testing/learning.
    
    Creates various synthetic signals: sine waves, chirps, noise, etc.
    """
    
    def __init__(self, num_samples=1000, sample_rate=16000, chunk_length=16000):
        """
        Args:
            num_samples: Number of synthetic audio samples to generate
            sample_rate: Sample rate
            chunk_length: Length of each audio chunk
        """
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.chunk_length = chunk_length
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Generate synthetic audio"""
        # Use idx as seed for reproducibility
        np.random.seed(idx)
        
        t = np.linspace(0, self.chunk_length / self.sample_rate, self.chunk_length)
        
        # Random audio type
        audio_type = idx % 5
        
        if audio_type == 0:
            # Sine wave
            freq = np.random.randint(100, 2000)
            audio = 0.5 * np.sin(2 * np.pi * freq * t)
        elif audio_type == 1:
            # Multiple frequencies (chord-like)
            freqs = [440, 550, 660]
            audio = np.zeros_like(t)
            for freq in freqs:
                audio += 0.3 * np.sin(2 * np.pi * freq * t)
        elif audio_type == 2:
            # Chirp (frequency sweep)
            f_start = 200
            f_end = 3000
            audio = np.sin(2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / 2))
        elif audio_type == 3:
            # White noise
            audio = np.random.randn(self.chunk_length) * 0.3
        else:
            # Speech-like: modulated noise
            envelope = 0.5 * (np.sin(2 * np.pi * t) + 1)
            audio = np.random.randn(self.chunk_length) * envelope * 0.3
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 1e-4:
            audio = audio / max_val
        
        audio = audio.astype(np.float32)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        
        return audio_tensor


class DataLoaderFactory:
    """Factory for creating data loaders"""
    
    @staticmethod
    def create_audio_loader(audio_dir, batch_size=4, sample_rate=16000, 
                           chunk_length=16000, shuffle=True, num_workers=0):
        """Create data loader from audio directory"""
        dataset = AudioDataset(audio_dir, sample_rate, chunk_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                         num_workers=num_workers)
    
    @staticmethod
    def create_synthetic_loader(num_samples=1000, batch_size=4, sample_rate=16000,
                               chunk_length=16000, shuffle=True):
        """Create data loader with synthetic audio"""
        dataset = SyntheticAudioDataset(num_samples, sample_rate, chunk_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class AudioNormalizer:
    """Normalize and denormalize audio"""
    
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
    
    def normalize(self, audio):
        """Normalize audio tensor"""
        return (audio - self.mean) / (self.std + 1e-8)
    
    def denormalize(self, audio):
        """Denormalize audio tensor"""
        return audio * self.std + self.mean
    
    @staticmethod
    def compute_stats(data_loader):
        """Compute statistics from data loader"""
        mean = 0.0
        std = 0.0
        num_batches = 0
        
        for batch in data_loader:
            mean += batch.mean().item()
            std += batch.std().item()
            num_batches += 1
        
        return mean / num_batches, std / num_batches
