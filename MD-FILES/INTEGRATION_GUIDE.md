# Neural Audio Compression - Integration Guide

> How to integrate Neural Audio Codec with your DEFLATE project

---

## Project Structure

Your project now has:

```
DEFLATE-fileCompressor/
├── neural_audio/                    ← NEW: Neural Audio Module
│   ├── __init__.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── quantizer.py
│   ├── model.py
│   ├── data_loader.py
│   ├── trainer.py
│   └── evaluator.py
│
├── huffman/                         ← Your existing Huffman
├── lz77/                           ← Your existing LZ77
├── utils/                          ← Your existing utils
├── visualization/                  ← Your existing visualization
│
├── compressor.py                   ← Your existing DEFLATE
├── decompressor.py                 ← Your existing decompressor
│
├── train_neural_codec.py           ← NEW: Training script
├── compress_audio_neural.py        ← NEW: Compression demo
├── examples_neural_audio.py        ← NEW: 7 learning examples
├── NEURAL_AUDIO_README.md          ← NEW: Detailed documentation
│
└── requirements.txt                ← UPDATED with torch, librosa, etc
```

---

## Step 1: Install Dependencies

```bash
bash
pip install -r requirements.txt
```

**Key packages added:**
- `torch==2.1.0` - PyTorch deep learning framework
- `librosa==0.10.0` - Audio processing
- `soundfile==0.12.1` - Audio I/O
- `tensorboard==2.14.1` - Training visualization

---

## Step 2: Train the Codec

### Option A: Quick Start (Synthetic Data)

```bash
python train_neural_codec.py \
  --epochs 50 \
  --batch-size 4 \
  --use-synthetic \
  --num-train-samples 1000
```

This trains on generated audio (sine waves, noise, chirps).

**Time:** ~5-30 minutes depending on GPU

### Option B: Train on Your Own Audio

```bash
# Create audio directory
mkdir audio_data

# Copy your audio files (.wav, .mp3, .flac)
cp /path/to/your/audio/*.wav audio_data/

# Train
python train_neural_codec.py \
  --use-synthetic=False \
  --audio-dir ./audio_data \
  --epochs 100 \
  --batch-size 8
```

### Monitoring Training

```bash
# View tensorboard logs
tensorboard --logdir ./neural_audio_checkpoints/logs
```

Then visit: `http://localhost:6006`

---

## Step 3: Use Pre-trained Codec

After training, you have checkpoints:

```
neural_audio_checkpoints/
├── logs/               ← Tensorboard logs
├── epoch_5.pt         ← Checkpoint at epoch 5
├── epoch_10.pt        ← Checkpoint at epoch 10
└── final_model.pt     ← Best model
```

---

## Step 4: Run Compression Demo

```bash
python compress_audio_neural.py
```

This will:
1. Create test audio if needed
2. Compress with Neural Codec
3. Compare with DEFLATE, LZMA, Zstandard
4. Generate:
   - `compressed_audio.json`
   - `reconstructed_audio.wav`

---

## Integration with Your DEFLATE Compressor

### Create Hybrid Compressor

```python
# hybrid_compressor.py
from compressor import Compressor  # Your DEFLATE
from neural_audio.model import NeuralAudioCodec
import torch

class HybridCompressor:
    """Intelligently choose compression method based on file type"""
    
    def __init__(self, checkpoint_path='neural_audio_checkpoints/final_model.pt'):
        self.deflate = Compressor()
        
        # Load pre-trained neural codec
        self.neural_codec = NeuralAudioCodec()
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.neural_codec.load_state_dict(checkpoint['model_state'])
            self.neural_codec.eval()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.neural_codec = self.neural_codec.to(self.device)
    
    def compress(self, filepath):
        """Compress file with appropriate method"""
        
        if filepath.endswith('.wav'):
            # Audio → Neural codec
            return self.compress_audio(filepath)
        else:
            # Regular → DEFLATE
            return self.compress_file(filepath)
    
    def compress_audio(self, audio_path):
        """Compress audio using neural codec"""
        import soundfile as sf
        import numpy as np
        
        # Load audio
        audio, sr = sf.read(audio_path)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        
        # To tensor
        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        
        # Compress
        with torch.no_grad():
            indices, latent_shape = self.neural_codec.encode(
                audio_tensor.to(self.device)
            )
        
        compression_ratio = self.neural_codec.get_compression_ratio()
        return {
            'method': 'Neural Codec',
            'ratio': compression_ratio,
            'indices': indices.cpu().numpy(),
        }
    
    def compress_file(self, file_path):
        """Compress regular file using DEFLATE"""
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Your DEFLATE compression
        encoded, root, tokens, codes = self.deflate.compress(data)
        
        compression_ratio = len(data) / len(encoded)
        return {
            'method': 'DEFLATE',
            'ratio': compression_ratio,
            'data': encoded,
        }
```

---

## Running Examples

Learn by example:

```bash
python examples_neural_audio.py
```

This runs 7 examples:
1. Basic codec usage
2. Encoder-decoder breakdown
3. Quantization in action
4. Data loading
5. Evaluation metrics
6. Mini training loop
7. Compression ratio math

---

## File-by-File Guide

### Training Script

```bash
# Full customization
python train_neural_codec.py \
  --epochs 100 \
  --batch-size 8 \
  --learning-rate 0.001 \
  --latent-dim 128 \
  --checkpoint-dir ./my_checkpoints \
  --chunk-length 16000 \
  --use-synthetic
```

**Output:**
```
neural_audio_checkpoints/
├── logs/                    ← Viewable in TensorBoard
│   ├── events.out.tfevents...
│   └── ...
├── epoch_5.pt
├── epoch_10.pt
└── final_model.pt           ← Use this
```

### Compression Script

```bash
python compress_audio_neural.py
```

Generates:
- `compressed_audio.json` - Compressed indices
- `reconstructed_audio.wav` - Decoded audio

---

## Key Components Reference

### Using the Codec

```python
from neural_audio.model import NeuralAudioCodec
import torch

# Create
codec = NeuralAudioCodec(latent_dim=128)

# Load trained model
checkpoint = torch.load('neural_audio_checkpoints/final_model.pt')
codec.load_state_dict(checkpoint['model_state'])
codec.eval()

# Compress
audio = torch.randn(1, 1, 16000)  # Your audio
with torch.no_grad():
    indices, shape = codec.encode(audio)

# Decompress
with torch.no_grad():
    reconstructed = codec.decode(indices, shape)

# Metrics
print(codec.get_compression_ratio())
print(codec.get_bitrate(duration_seconds=1.0))
```

### Training from Scratch

```python
from neural_audio.trainer import AudioCompressionTrainer
from neural_audio.data_loader import DataLoaderFactory

# Setup
codec = NeuralAudioCodec()
trainer = AudioCompressionTrainer(codec, device='cuda')
trainer.setup_optimizer()

# Data
loader = DataLoaderFactory.create_synthetic_loader(
    num_samples=1000,
    batch_size=4
)

# Train
trainer.train(loader, epochs=100, save_interval=5)
```

### Evaluation

```python
from neural_audio.evaluator import AudioEvaluator

evaluator = AudioEvaluator(sample_rate=16000)

# Evaluate batch
metrics = evaluator.evaluate_reconstruction(original, reconstructed)
print(f"MSE: {metrics['mse']}")
print(f"SNR: {metrics['snr_db']} dB")

# Save results
evaluator.save_results('results.json')
```

---

## Troubleshooting

### ImportError: No module named 'torch'

```bash
pip install torch torchaudio librosa soundfile
```

### CUDA out of memory

```python
# Reduce batch size
batch_size = 2  # instead of 8

# Reduce latent dimension  
latent_dim = 64  # instead of 128

# Reduce chunk length
chunk_length = 8000  # instead of 16000
```

### Training loss not decreasing

```python
# Try lower learning rate
learning_rate = 0.0001

# More diverse data helps
num_samples = 5000  # instead of 1000

# Train longer
epochs = 200  # instead of 50
```

### Model not loading

```python
# Make sure checkpoint path is correct
checkpoint = torch.load('neural_audio_checkpoints/final_model.pt')

# Check device matching
codec = codec.to('cuda')
checkpoint = torch.load(path, map_location='cuda')
```

---

## Next: Advanced Features

Once basic training works, try:

### 1. Fast Fine-Tuning

```python
from neural_audio.trainer import FineTuningTrainer

# Load pre-trained
codec = load_pretrained_codec()

trainer = FineTuningTrainer(codec)
trainer.freeze_encoder()  # Only train decoder
trainer.train(new_data, epochs=10)
```

### 2. Knowledge Distillation

```python
from neural_audio.trainer import DistillationTrainer

# Large teacher, small student
teacher = load_large_codec()
student = NeuralAudioCodec(latent_dim=64)

trainer = DistillationTrainer(student, teacher)
trainer.train(data, epochs=100)
```

### 3. Multi-Rate Codec

```python
from neural_audio.model import MultiRateCodec

codec = MultiRateCodec(num_qualities=4)

# Quality 0: Maximum compression
# Quality 3: Maximum quality
audio_q0 = codec(audio, quality=0)
audio_q3 = codec(audio, quality=3)
```

---

## Recommended Workflow

### Week 1: Learn
- Run `examples_neural_audio.py`
- Read `NEURAL_AUDIO_README.md`
- Understand each component

### Week 2-3: Train
- Train on synthetic data (quick, no GPU required)
- Collect real audio for better results
- Monitor training with TensorBoard
- Save checkpoints

### Week 4: Integrate
- Use trained model in your app
- Compare with DEFLATE
- Create visualizations
- Document results

### Week 5+: Polish
- Fine-tune for your use case
- Optimize for speed
- Add to main compressor
- Create GitHub showcase

---

## Resume Points

This implementation shows:

✅ **Deep Learning**: CNN architecture, loss functions, training loop
✅ **Audio Processing**: Signal processing, sampling, quantization
✅ **Research Implementation**: Based on real papers (VQVAE, SoundStream)
✅ **Production Code**: Proper abstractions, error handling, documentation
✅ **Benchmarking**: Comparison with existing methods
✅ **MLOps**: Checkpointing, TensorBoard, evaluation metrics

---

## Questions?

See `NEURAL_AUDIO_README.md` for detailed documentation of every component.

Good luck! 🚀
