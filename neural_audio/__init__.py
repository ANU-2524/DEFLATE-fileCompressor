"""
Neural Audio Compression Module
Implements deep learning-based audio compression with encoder-decoder architecture
"""

from .model import NeuralAudioCodec
from .trainer import AudioCompressionTrainer
from .encoder import AudioEncoder
from .decoder import AudioDecoder
from .quantizer import VectorQuantizer
from .data_loader import AudioDataset, DataLoaderFactory
from .evaluator import AudioEvaluator

__all__ = [
    'NeuralAudioCodec',
    'AudioCompressionTrainer',
    'AudioEncoder',
    'AudioDecoder',
    'VectorQuantizer',
    'AudioDataset',
    'DataLoaderFactory',
    'AudioEvaluator',
]
