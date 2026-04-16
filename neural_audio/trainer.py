"""
Training Pipeline for Neural Audio Codec
"""

import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


class AudioCompressionTrainer:
    """
    Trainer for neural audio codec.
    
    Handles training loop, validation, checkpointing, and logging.
    
    Example:
        trainer = AudioCompressionTrainer(codec, device='cuda')
        trainer.train(train_loader, val_loader, epochs=100)
    """
    
    def __init__(self, model, device='cpu', checkpoint_dir='./checkpoints'):
        """
        Args:
            model: NeuralAudioCodec instance
            device: 'cpu' or 'cuda'
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Logging
        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / 'logs'))
        self.train_losses = []
        self.val_losses = []
        self.epoch = 0
    
    def setup_optimizer(self, lr=1e-3, weight_decay=1e-5):
        """Setup optimizer"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        return self.optimizer
    
    def train_epoch(self, train_loader, loss_weights=None):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            loss_weights: Dict with loss weights
                - 'vq': weight for quantization loss
                - 'recon': weight for reconstruction loss
        
        Returns:
            avg_loss: Average loss for epoch
        """
        if loss_weights is None:
            loss_weights = {'vq': 0.25, 'recon': 1.0}
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, audio in enumerate(train_loader):
            audio = audio.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed, losses = self.model(audio)
            
            # Calculate total loss
            vq_loss = losses['vq_loss']
            recon_loss = losses['reconstruction_loss']
            total = loss_weights['vq'] * vq_loss + loss_weights['recon'] * recon_loss
            
            # Backward pass
            total.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total.item()
            num_batches += 1
            
            # Log to tensorboard
            global_step = self.epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Loss/vq', vq_loss.item(), global_step)
            self.writer.add_scalar('Loss/reconstruction', recon_loss.item(), global_step)
            self.writer.add_scalar('Loss/total', total.item(), global_step)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {self.epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {total.item():.4f}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader):
        """
        Validation pass without gradients.
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            avg_loss: Average validation loss
            metrics: Dict of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for audio in val_loader:
                audio = audio.to(self.device)
                
                # Forward pass
                reconstructed, losses = self.model(audio)
                vq_loss = losses['vq_loss']
                recon_loss = losses['reconstruction_loss']
                total = 0.25 * vq_loss + recon_loss
                
                total_loss += total.item()
                total_recon_loss += recon_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        
        self.val_losses.append(avg_loss)
        
        metrics = {
            'total_loss': avg_loss,
            'reconstruction_loss': avg_recon_loss,
        }
        
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader=None, epochs=100, save_interval=5):
        """
        Complete training loop.
        
        Args:
            train_loader: DataLoader for training
            val_loader: Optional DataLoader for validation
            epochs: Number of epochs to train
            save_interval: Save checkpoint every N epochs
        """
        if self.optimizer is None:
            self.setup_optimizer()
        
        print(f"Training on {self.device}")
        print(f"Total epochs: {epochs}")
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Training
            train_loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
            
            # Validation
            if val_loader is not None:
                val_loss, metrics = self.validate(val_loader)
                self.writer.add_scalar('Validation/loss', val_loss, epoch)
                print(f"Validation Loss: {val_loss:.4f}")
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
            
            # Checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pt")
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        print("Training complete!")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        if self.optimizer and checkpoint['optimizer_state']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"Loaded checkpoint from epoch {self.epoch}")


class FineTuningTrainer(AudioCompressionTrainer):
    """
    Trainer for fine-tuning a pre-trained codec.
    
    Allows selective freezing of components.
    """
    
    def __init__(self, model, device='cpu', checkpoint_dir='./checkpoints'):
        super().__init__(model, device, checkpoint_dir)
    
    def freeze_encoder(self):
        """Freeze encoder, train only decoder"""
        self.model.freeze_encoder()
        print("Encoder frozen - training decoder only")
    
    def freeze_decoder(self):
        """Freeze decoder, train only encoder"""
        self.model.freeze_decoder()
        print("Decoder frozen - training encoder only")
    
    def unfreeze_all(self):
        """Unfreeze all components"""
        self.model.unfreeze_all()
        print("All components unfrozen")


class DistillationTrainer(AudioCompressionTrainer):
    """
    Trainer for knowledge distillation - train smaller student codec
    from larger teacher codec.
    """
    
    def __init__(self, student_model, teacher_model, device='cpu'):
        super().__init__(student_model, device)
        self.teacher_model = teacher_model.to(device)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
    
    def train_epoch(self, train_loader, loss_weights=None, temperature=3.0):
        """Train with knowledge distillation"""
        if loss_weights is None:
            loss_weights = {'vq': 0.25, 'recon': 0.8, 'distill': 0.2}
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, audio in enumerate(train_loader):
            audio = audio.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Student forward pass
            student_recon, student_losses = self.model(audio)
            
            # Teacher forward pass (no grad)
            with torch.no_grad():
                teacher_recon, _ = self.teacher_model(audio)
            
            # Losses
            vq_loss = student_losses['vq_loss']
            recon_loss = student_losses['reconstruction_loss']
            distill_loss = torch.nn.functional.mse_loss(student_recon, teacher_recon.detach())
            
            total = (loss_weights['vq'] * vq_loss + 
                    loss_weights['recon'] * recon_loss +
                    loss_weights['distill'] * distill_loss)
            
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
