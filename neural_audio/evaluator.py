"""
Evaluation Metrics for Neural Audio Compression
"""

import numpy as np
import torch
import json
from pathlib import Path


class AudioEvaluator:
    """
    Evaluate audio compression quality and efficiency.
    
    Metrics:
    - Reconstruction Error (MSE, MAE)
    - Perceptual Quality (PESQ, STOI - if available)
    - Compression Ratio
    - Bitrate
    - Encoding/Decoding Speed
    """
    
    def __init__(self, sample_rate=16000):
        """
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.results = []
    
    def evaluate_reconstruction(self, original, reconstructed):
        """
        Evaluate reconstruction quality.
        
        Args:
            original: [batch, channels, samples]
            reconstructed: [batch, channels, samples]
        
        Returns:
            metrics: Dict with error metrics
        """
        original = original.cpu().numpy() if isinstance(original, torch.Tensor) else original
        reconstructed = reconstructed.cpu().numpy() if isinstance(reconstructed, torch.Tensor) else reconstructed
        
        # Ensure same shape
        min_len = min(original.shape[-1], reconstructed.shape[-1])
        original = original[..., :min_len]
        reconstructed = reconstructed[..., :min_len]
        
        # Calculate errors
        mse = np.mean((original - reconstructed) ** 2)
        mae = np.mean(np.abs(original - reconstructed))
        rmse = np.sqrt(mse)
        
        # Signal to Noise Ratio (SNR)
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - reconstructed) ** 2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Segmental SNR
        frame_len = int(0.02 * self.sample_rate)  # 20ms frames
        seg_snr = []
        for i in range(0, len(original) - frame_len, frame_len):
            sig = original[..., i:i+frame_len]
            noise = original[..., i:i+frame_len] - reconstructed[..., i:i+frame_len]
            seg_snr.append(10 * np.log10(np.mean(sig**2) / (np.mean(noise**2) + 1e-10)))
        seg_snr_db = np.mean(seg_snr) if seg_snr else 0
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'snr_db': float(snr_db),
            'seg_snr_db': float(seg_snr_db),
        }
        
        return metrics
    
    def spectral_distance(self, original, reconstructed):
        """
        Calculate spectral distance using STFT.
        
        Measures difference in frequency domain.
        
        Args:
            original: [batch, channels, samples]
            reconstructed: [batch, channels, samples]
        
        Returns:
            distance: Float
        """
        try:
            import librosa
            
            original = original.cpu().numpy().flatten() if isinstance(original, torch.Tensor) else original.flatten()
            reconstructed = reconstructed.cpu().numpy().flatten() if isinstance(reconstructed, torch.Tensor) else reconstructed.flatten()
            
            # Compute STFT
            D_orig = librosa.stft(original)
            D_recon = librosa.stft(reconstructed)
            
            # Magnitude spectrogram
            S_orig = np.abs(D_orig)
            S_recon = np.abs(D_recon)
            
            # Log scaling
            S_orig = librosa.power_to_db(S_orig**2 + 1e-10, ref=np.max)
            S_recon = librosa.power_to_db(S_recon**2 + 1e-10, ref=np.max)
            
            # Spectral distance (average L2 distance)
            distance = np.mean(np.sqrt(np.sum((S_orig - S_recon)**2, axis=0)))
            
            return float(distance)
        except ImportError:
            print("librosa not available - skipping spectral distance")
            return None
    
    def compression_metrics(self, original_size, compressed_size):
        """
        Calculate compression efficiency metrics.
        
        Args:
            original_size: Original size in bytes
            compressed_size: Compressed size in bytes
        
        Returns:
            metrics: Dict with compression metrics
        """
        compression_ratio = original_size / (compressed_size + 1e-10)
        compression_percent = (1 - compressed_size / original_size) * 100
        
        metrics = {
            'compression_ratio': float(compression_ratio),
            'compression_percent': float(compression_percent),
            'original_size_bytes': int(original_size),
            'compressed_size_bytes': int(compressed_size),
        }
        
        return metrics
    
    def bitrate_metrics(self, compressed_size, duration_seconds):
        """
        Calculate bitrate metrics.
        
        Args:
            compressed_size: Size in bytes
            duration_seconds: Duration in seconds
        
        Returns:
            metrics: Dict with bitrate metrics
        """
        total_bits = compressed_size * 8
        bitrate_kbps = (total_bits / 1000) / duration_seconds
        bitrate_mbps = bitrate_kbps / 1000
        
        metrics = {
            'bitrate_kbps': float(bitrate_kbps),
            'bitrate_mbps': float(bitrate_mbps),
        }
        
        return metrics
    
    def try_pesq_score(self, original, reconstructed):
        """
        Calculate PESQ score if available.
        
        PESQ = Perceptual Evaluation of Speech Quality
        Range: -0.5 to 4.5 (higher is better)
        
        Args:
            original: [batch, channels, samples]
            reconstructed: [batch, channels, samples]
        
        Returns:
            score: Float or None if not available
        """
        try:
            from pesq import pesq
            
            original = original.cpu().numpy().flatten() if isinstance(original, torch.Tensor) else original.flatten()
            reconstructed = reconstructed.cpu().numpy().flatten() if isinstance(reconstructed, torch.Tensor) else reconstructed.flatten()
            
            # Ensure same length
            min_len = min(len(original), len(reconstructed))
            original = original[:min_len]
            reconstructed = reconstructed[:min_len]
            
            # PESQ requires 16-bit audio
            original = (original * 32767).astype(np.int16)
            reconstructed = (reconstructed * 32767).astype(np.int16)
            
            score = pesq(self.sample_rate, original, reconstructed, mode='wb')
            return float(score)
        except ImportError:
            return None
    
    def try_stoi_score(self, original, reconstructed):
        """
        Calculate STOI score if available.
        
        STOI = Short-Time Objective Intelligibility
        Range: 0 to 1 (higher is better)
        
        Args:
            original: [batch, channels, samples]
            reconstructed: [batch, channels, samples]
        
        Returns:
            score: Float or None if not available
        """
        try:
            from pystoi import stoi
            
            original = original.cpu().numpy().flatten() if isinstance(original, torch.Tensor) else original.flatten()
            reconstructed = reconstructed.cpu().numpy().flatten() if isinstance(reconstructed, torch.Tensor) else reconstructed.flatten()
            
            score = stoi(original, reconstructed, self.sample_rate)
            return float(score)
        except ImportError:
            return None
    
    def evaluate_batch(self, original, reconstructed, original_size=None, compressed_size=None):
        """
        Comprehensive evaluation of a batch.
        
        Args:
            original: Original audio tensor
            reconstructed: Reconstructed audio tensor
            original_size: Original size in bytes (optional)
            compressed_size: Compressed size in bytes (optional)
        
        Returns:
            results: Dict with all metrics
        """
        results = {}
        
        # Reconstruction metrics
        recon_metrics = self.evaluate_reconstruction(original, reconstructed)
        results.update(recon_metrics)
        
        # Spectral metrics
        spec_dist = self.spectral_distance(original, reconstructed)
        if spec_dist is not None:
            results['spectral_distance'] = spec_dist
        
        # Perceptual metrics
        pesq = self.try_pesq_score(original, reconstructed)
        if pesq is not None:
            results['pesq_score'] = pesq
        
        stoi = self.try_stoi_score(original, reconstructed)
        if stoi is not None:
            results['stoi_score'] = stoi
        
        # Compression metrics
        if original_size and compressed_size:
            comp_metrics = self.compression_metrics(original_size, compressed_size)
            results.update(comp_metrics)
            
            # Bitrate metrics
            duration = original.shape[-1] / self.sample_rate
            bitrate_metrics = self.bitrate_metrics(compressed_size, duration)
            results.update(bitrate_metrics)
        
        self.results.append(results)
        return results
    
    def get_summary(self):
        """Get summary statistics across all evaluations"""
        if not self.results:
            return {}
        
        summary = {}
        for key in self.results[0].keys():
            values = [r[key] for r in self.results if key in r]
            if values:
                summary[f'{key}_mean'] = float(np.mean(values))
                summary[f'{key}_std'] = float(np.std(values))
                summary[f'{key}_min'] = float(np.min(values))
                summary[f'{key}_max'] = float(np.max(values))
        
        return summary
    
    def print_summary(self):
        """Print detailed summary"""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        for key, value in sorted(summary.items()):
            print(f"{key:.<40} {value:.4f}")
        print("="*60)
    
    def save_results(self, filepath):
        """Save results to JSON file"""
        data = {
            'all_results': self.results,
            'summary': self.get_summary(),
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {filepath}")


class CompressionBenchmark:
    """
    Benchmark codec against competition (Opus, MP3, etc.)
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.evaluator = AudioEvaluator(sample_rate)
        self.benchmarks = {}
    
    def add_result(self, codec_name, metrics):
        """Add benchmark result"""
        self.benchmarks[codec_name] = metrics
    
    def compare(self, original, neural_recon, opus_recon=None, mp3_recon=None):
        """
        Compare neural codec with traditional codecs.
        
        Args:
            original: Original audio
            neural_recon: Reconstruction from neural codec
            opus_recon: Reconstruction from Opus (optional)
            mp3_recon: Reconstruction from MP3 (optional)
        
        Returns:
            comparison: Dict with comparison results
        """
        comparison = {}
        
        # Neural codec
        neural_metrics = self.evaluator.evaluate_reconstruction(original, neural_recon)
        comparison['Neural Audio Codec'] = neural_metrics
        
        # Opus
        if opus_recon is not None:
            opus_metrics = self.evaluator.evaluate_reconstruction(original, opus_recon)
            comparison['Opus'] = opus_metrics
        
        # MP3
        if mp3_recon is not None:
            mp3_metrics = self.evaluator.evaluate_reconstruction(original, mp3_recon)
            comparison['MP3'] = mp3_metrics
        
        return comparison
    
    def print_comparison(self, comparison):
        """Pretty print comparison"""
        print("\n" + "="*70)
        print("CODEC COMPARISON")
        print("="*70)
        
        # Get all metric names
        metrics = set()
        for codec_results in comparison.values():
            metrics.update(codec_results.keys())
        
        metrics = sorted(metrics)
        codecs = list(comparison.keys())
        
        # Print header
        print(f"{'Metric':<20}", end='')
        for codec in codecs:
            print(f"{codec:>15}", end='')
        print()
        print("-"*70)
        
        # Print rows
        for metric in metrics:
            print(f"{metric:<20}", end='')
            for codec in codecs:
                value = comparison[codec].get(metric, 0)
                print(f"{value:>15.4f}", end='')
            print()
        
        print("="*70)
