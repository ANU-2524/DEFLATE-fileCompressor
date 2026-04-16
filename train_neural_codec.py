"""
Training Script for Neural Audio Compression
Run this to train the neural codec from scratch
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_audio.model import NeuralAudioCodec
from neural_audio.trainer import AudioCompressionTrainer
from neural_audio.data_loader import DataLoaderFactory, SyntheticAudioDataset
from neural_audio.evaluator import AudioEvaluator


def train_neural_codec(args):
    """Main training function"""
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    codec = NeuralAudioCodec(
        latent_dim=args.latent_dim,
        num_channels=1,
        num_codebook_entries=256,  # 8 bits per index
        sample_rate=args.sample_rate
    )
    
    # Create trainer
    trainer = AudioCompressionTrainer(
        codec,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Setup optimizer
    trainer.setup_optimizer(lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Create data loaders
    print("Creating data loaders...")
    if args.use_synthetic:
        print(f"Using synthetic audio data ({args.num_train_samples} samples)")
        train_loader = DataLoaderFactory.create_synthetic_loader(
            num_samples=args.num_train_samples,
            batch_size=args.batch_size,
            sample_rate=args.sample_rate,
            chunk_length=args.chunk_length,
            shuffle=True
        )
        
        val_loader = DataLoaderFactory.create_synthetic_loader(
            num_samples=args.num_val_samples,
            batch_size=args.batch_size,
            sample_rate=args.sample_rate,
            chunk_length=args.chunk_length,
            shuffle=False
        ) if args.num_val_samples > 0 else None
    else:
        print(f"Loading audio from {args.audio_dir}")
        train_loader = DataLoaderFactory.create_audio_loader(
            args.audio_dir,
            batch_size=args.batch_size,
            sample_rate=args.sample_rate,
            chunk_length=args.chunk_length,
            shuffle=True
        )
        val_loader = None  # Can add validation directory if needed
    
    # Train
    print("\nStarting training...")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Chunk length: {args.chunk_length} samples ({args.chunk_length/args.sample_rate:.2f}s)")
    print()
    
    trainer.train(
        train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_interval=args.save_interval
    )
    
    # Final evaluation on training data
    print("\nFinal evaluation...")
    evaluator = AudioEvaluator(sample_rate=args.sample_rate)
    
    codec.eval()
    with torch.no_grad():
        for batch_idx, audio in enumerate(train_loader):
            if batch_idx >= 5:  # Evaluate on first 5 batches
                break
            
            audio = audio.to(device)
            reconstructed, losses = codec(audio)
            
            # Evaluate
            metrics = evaluator.evaluate_reconstruction(audio, reconstructed)
            print(f"Batch {batch_idx}: MSE={metrics['mse']:.6f}, SNR={metrics['snr_db']:.2f}dB")
    
    evaluator.print_summary()
    
    print("\nTraining complete!")
    print(f"Checkpoint directory: {args.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Neural Audio Codec')
    
    # Model arguments
    parser.add_argument('--latent-dim', type=int, default=128,
                       help='Latent dimension size (default: 128)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Audio sample rate (default: 16000)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay (default: 1e-5)')
    parser.add_argument('--save-interval', type=int, default=5,
                       help='Save checkpoint every N epochs (default: 5)')
    
    # Data arguments
    parser.add_argument('--use-synthetic', action='store_true', default=True,
                       help='Use synthetic audio data (default: True)')
    parser.add_argument('--num-train-samples', type=int, default=1000,
                       help='Number of synthetic training samples (default: 1000)')
    parser.add_argument('--num-val-samples', type=int, default=200,
                       help='Number of synthetic validation samples (default: 200)')
    parser.add_argument('--audio-dir', type=str, default='./audio_data',
                       help='Directory with real audio files (default: ./audio_data)')
    parser.add_argument('--chunk-length', type=int, default=16000,
                       help='Audio chunk length in samples (default: 16000 = 1 second @ 16kHz)')
    
    # Checkpoint directory
    parser.add_argument('--checkpoint-dir', type=str, default='./neural_audio_checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    train_neural_codec(args)


if __name__ == '__main__':
    main()
