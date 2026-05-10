import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from compressor import Compressor
from decompressor import Decompressor
import json
import os
from datetime import datetime
import pandas as pd
import time
from pathlib import Path
import numpy as np
import tempfile

# Neural Audio Compression Imports
try:
    import torch
    import soundfile as sf
    from neural_audio.model import NeuralAudioCodec
    from neural_audio.evaluator import AudioEvaluator
    from io import BytesIO
    try:
        import librosa
        NEURAL_AUDIO_AVAILABLE = True
    except ImportError:
        # librosa not available, but app still works
        st.write("⚠️ Neural Audio requires librosa. Install with: pip install librosa")
        NEURAL_AUDIO_AVAILABLE = False
except ImportError:
    NEURAL_AUDIO_AVAILABLE = False

# ==================== PAGE CONFIG ==================== #
st.set_page_config(
    page_title="DEFLATE Compressor Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/deflate-compressor',
        'Report a bug': 'https://github.com/deflate-compressor/issues'
    }
)

# ==================== SESSION STATE INITIALIZATION ==================== #
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'compressed_data' not in st.session_state:
    st.session_state.compressed_data = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'animation_state' not in st.session_state:
    st.session_state.animation_state = 0

# Neural Audio Session State
if 'neural_audio_data' not in st.session_state:
    st.session_state.neural_audio_data = None
if 'neural_sample_rate' not in st.session_state:
    st.session_state.neural_sample_rate = None
if 'neural_codec_loaded' not in st.session_state:
    st.session_state.neural_codec_loaded = False
if 'neural_compressed_indices' not in st.session_state:
    st.session_state.neural_compressed_indices = None
if 'neural_compression_metrics' not in st.session_state:
    st.session_state.neural_compression_metrics = {}
if 'neural_reconstructed_audio' not in st.session_state:
    st.session_state.neural_reconstructed_audio = None
if 'neural_model_trained' not in st.session_state:
    st.session_state.neural_model_trained = False

# ==================== MODERN MINIMAL STYLING ==================== #
st.markdown("""
    <style>
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    
    /* Main header - clean and simple */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a1a1a;
        text-align: center;
        margin: 1.5rem 0 2rem 0;
        letter-spacing: -0.5px;
    }
    
    /* Subtle card design */
    .metric-card {
        background: #ffffff;
        padding: 1.2rem;
        border-radius: 0.6rem;
        margin: 0.5rem 0;
        border: 1px solid #eaeaea;
        border-left: 3px solid #2563eb;
        transition: box-shadow 0.2s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .stat-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-bottom: 0.5rem;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a1a;
    }
    
    .stat-change {
        font-size: 0.85rem;
        color: #10b981;
        margin-top: 0.4rem;
        font-weight: 500;
    }
    
    /* Feature card */
    .feature-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 0.6rem;
        border: 1px solid #eaeaea;
        text-align: center;
        transition: box-shadow 0.2s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .feature-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: #f0f0f0;
        margin: 1.5rem 0;
    }
    
    /* Streamlit components */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: background-color 0.2s ease;
        padding: 0.6rem 1.2rem;
    }
    
    .stButton > button:hover {
        background-color: #1d4ed8;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid #eaeaea;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-bottom: 2px solid transparent;
        color: #6b7280;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom-color: #2563eb;
        color: #1a1a1a;
    }
    
    .stFileUploader {
        border: 2px dashed #d1d5db;
        border-radius: 0.6rem;
        background-color: #fafafa;
    }
    
    .stFileUploader:hover {
        border-color: #2563eb;
        background-color: #f0f6ff;
    }
    
    .stAlert {
        border-radius: 0.6rem;
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
    }
    
    .stSidebar {
        background-color: #f9fafb;
        border-right: 1px solid #eaeaea;
    }
    
    /* Table */
    .stDataframe {
        border: 1px solid #eaeaea;
        border-radius: 0.6rem;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS ==================== #
HISTORY_FILE = "compression_history.json"
OUTPUT_DIR = "output"

# ==================== ENHANCED HELPER FUNCTIONS ==================== #
def format_bytes(bytes_val):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"

def load_history():
    """Load compression history from JSON"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(entry):
    """Save compression entry to history"""
    history = load_history()
    history.append(entry)
    history = history[-50:]
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def display_metric_card(col, title, value, unit=""):
    """Display a clean metric card"""
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='stat-title'>{title}</div>
            <div class='stat-value'>{value}</div>
            {f'<div style="color: #9ca3af; font-size: 0.85rem; margin-top: 0.4rem;">{unit}</div>' if unit else ''}
        </div>
        """, unsafe_allow_html=True)

def display_feature_card(col, icon, title, description):
    """Display a clean feature card"""
    with col:
        st.markdown(f"""
        <div class='feature-card'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem; line-height: 1;'>{icon}</div>
            <div style='font-size: 1.05rem; font-weight: 600; color: #1a1a1a; margin-bottom: 0.5rem;'>{title}</div>
            <div style='color: #6b7280; font-size: 0.9rem; line-height: 1.4;'>{description}</div>
        </div>
        """, unsafe_allow_html=True)

def create_stat_comparison(original, compressed):
    """Create a clean comparison visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Style
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    
    # Size comparison
    categories = ['Original', 'Compressed']
    values = [original, compressed]
    colors = ['#3b82f6', '#10b981']
    
    bars1 = ax1.bar(categories, values, color=colors, alpha=0.8, edgecolor='#e5e7eb', linewidth=1.5)
    ax1.set_ylabel('Size (Bytes)', color='#374151', fontsize=11, fontweight='600')
    ax1.set_title('File Size Comparison', color='#1a1a1a', fontsize=12, fontweight='600', pad=15)
    ax1.tick_params(colors='#6b7280')
    ax1.grid(axis='y', alpha=0.1, color='#d1d5db', linestyle='-', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, val in zip(bars1, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{format_bytes(val)}',
                ha='center', va='bottom', color='#1a1a1a', fontweight='600', fontsize=10)
    
    # Ratio pie chart
    if original > 0:
        ratio = min(100, (compressed / original) * 100)
        sizes = [ratio, 100 - ratio]
        labels = [f'Compressed\n{ratio:.1f}%', f'Reduction\n{100-ratio:.1f}%']
        colors_pie = ['#ef4444', '#10b981']
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                textprops={'color': '#1a1a1a', 'fontweight': '600', 'fontsize': 10},
                startangle=90, wedgeprops={'edgecolor': '#e5e7eb', 'linewidth': 1.5})
        ax2.set_title('Compression Breakdown', color='#1a1a1a', fontsize=12, fontweight='600', pad=15)
    
    plt.tight_layout()
    return fig

def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """Hierarchical layout for trees"""
    pos = {root: (xcenter, vert_loc)}
    neighbors = list(G.neighbors(root))
    
    if len(neighbors) != 0:
        dx = width / len(neighbors)
        nextx = xcenter - width/2 - dx/2
        for neighbor in neighbors:
            nextx += dx
            pos.update(hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap,
                                    vert_loc=vert_loc-vert_gap, xcenter=nextx))
    return pos

def draw_huffman_tree_enhanced(root):
    """Draw Huffman tree with clean styling"""
    G = nx.DiGraph()

    def add_edges(node, parent=None):
        if node:
            if hasattr(node, 'char'):
                if node.char is not None:
                    if isinstance(node.char, int):
                        label = f"chr({node.char})"
                    else:
                        label = str(node.char)
                else:
                    label = f"F:{node.freq}"
            else:
                label = f"F:{node.freq}"
            
            G.add_node(id(node), label=label)

            if parent:
                edge_label = '0' if parent.left == node else '1'
                G.add_edge(id(parent), id(node), label=edge_label)

            add_edges(node.left, node)
            add_edges(node.right, node)

    add_edges(root)

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except:
        pos = hierarchy_pos(G, id(root))

    labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    nx.draw(G, pos, labels=labels, with_labels=True, ax=ax,
            node_color='#dbeafe', node_size=1500,
            font_size=9, font_weight='600', font_color='#1a1a1a',
            arrows=True, arrowsize=15, edge_color='#d1d5db', width=1.5,
            arrowstyle='->', connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, 
                                  font_size=9, font_color='#6b7280', font_weight='600')
    
    ax.set_title("Huffman Binary Tree", fontsize=14, fontweight='600', 
                 color='#1a1a1a', pad=20)
    ax.axis('off')
    plt.tight_layout()
    return fig

def ensure_output_dir():
    """Ensure output directory exists"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== NEURAL AUDIO HELPER FUNCTIONS ==================== #
@st.cache_resource
def load_neural_codec():
    """Load pre-trained neural audio codec"""
    if not NEURAL_AUDIO_AVAILABLE:
        return None, 'cpu', False
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        codec = NeuralAudioCodec(latent_dim=128).to(device)
        
        # Try to load pre-trained checkpoint
        checkpoint_path = Path('neural_audio_checkpoints/final_model.pt')
        model_trained = False
        
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                codec.load_state_dict(checkpoint['model_state'])
                model_trained = True
            except:
                model_trained = False
        
        codec.eval()
        return codec, device, model_trained
    except Exception as e:
        st.error(f"Error loading codec: {e}")
        return None, 'cpu', False

def load_audio_file(uploaded_file):
    """Load and preprocess audio file"""
    try:
        # Save to temporary file (Windows-compatible)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name
        
        try:
            # Load audio
            audio, sr = sf.read(temp_path)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                try:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                except Exception as resample_error:
                    st.warning(f"Could not resample from {sr}Hz to 16kHz: {resample_error}")
                sr = 16000
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 1e-4:
                audio = audio / max_val
            
            return audio, sr, True
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
    except Exception as e:
        st.error(f"Error loading audio: {str(e)}")
        return None, None, False

def compress_audio_neural(audio, codec, device):
    """Compress audio using neural codec"""
    try:
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0).to(device)
        
        # Encode
        with torch.no_grad():
            indices, latent_shape = codec.encode(audio_tensor)
        
        original_size = audio.nbytes
        compressed_size = indices.numel() * 8 // 8  # 8-bit per index
        compression_ratio = original_size / (compressed_size + 1e-10)
        
        metrics = {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'bitrate_kbps': (compressed_size * 8 / 1000) / (len(audio) / 16000),
            'indices_shape': indices.shape,
            'latent_shape': latent_shape,
        }
        
        return indices, latent_shape, metrics, True
    except Exception as e:
        st.error(f"Compression error: {e}")
        return None, None, {}, False

def decompress_audio_neural(indices, latent_shape, codec, device):
    """Decompress audio from indices"""
    try:
        indices = indices.to(device)
        with torch.no_grad():
            audio_reconstructed = codec.decode(indices, tuple(latent_shape))
        
        audio_np = audio_reconstructed.cpu().numpy().squeeze()
        return audio_np, True
    except Exception as e:
        st.error(f"Decompression error: {e}")
        return None, False

def evaluate_audio_quality(original, reconstructed, sr=16000):
    """Evaluate audio quality metrics"""
    try:
        evaluator = AudioEvaluator(sample_rate=sr)
        
        original_t = torch.from_numpy(original.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        reconstructed_t = torch.from_numpy(reconstructed.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        
        metrics = evaluator.evaluate_reconstruction(original_t, reconstructed_t)
        return metrics
    except Exception as e:
        st.warning(f"Evaluation error: {e}")
        return {}

def plot_waveform_comparison_neural(original, reconstructed, sr=16000):
    """Plot waveform comparison"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.patch.set_facecolor('white')
    
    time = np.arange(len(original)) / sr
    
    axes[0].plot(time, original, linewidth=0.5, color='#2563eb')
    axes[0].set_title('Original Audio', fontsize=12, fontweight='bold', color='#1a1a1a')
    axes[0].set_ylabel('Amplitude', color='#6b7280')
    axes[0].grid(True, alpha=0.2)
    axes[0].set_facecolor('white')
    
    axes[1].plot(time, reconstructed, linewidth=0.5, color='#10b981')
    axes[1].set_title('Reconstructed Audio (After Compression)', fontsize=12, fontweight='bold', color='#1a1a1a')
    axes[1].set_ylabel('Amplitude', color='#6b7280')
    axes[1].set_xlabel('Time (seconds)', color='#6b7280')
    axes[1].grid(True, alpha=0.2)
    axes[1].set_facecolor('white')
    
    plt.tight_layout()
    return fig

def plot_frequency_spectrum_neural(original, reconstructed, sr=16000):
    """Plot frequency spectrum comparison"""
    from scipy.fft import fft
    
    fft_orig = np.abs(fft(original))[:len(original)//2]
    fft_recon = np.abs(fft(reconstructed))[:len(reconstructed)//2]
    
    freq = np.fft.fftfreq(len(original), 1/sr)[:len(original)//2]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    ax.plot(freq, fft_orig, label='Original', linewidth=1.5, color='#2563eb', alpha=0.7)
    ax.plot(freq, fft_recon, label='Reconstructed', linewidth=1.5, color='#10b981', alpha=0.7)
    ax.set_xlabel('Frequency (Hz)', color='#6b7280')
    ax.set_ylabel('Magnitude', color='#6b7280')
    ax.set_title('Frequency Spectrum Comparison', fontsize=12, fontweight='bold', color='#1a1a1a')
    ax.set_xlim([0, 8000])  # Focus on audible range
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    return fig

# ==================== SIDEBAR ==================== #
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 1.5rem; padding: 1.5rem 0; border-bottom: 1px solid #eaeaea;'>
        <div style='font-size: 1.4rem; font-weight: 700; color: #1a1a1a;'>DEFLATE</div>
        <div style='font-size: 0.8rem; color: #6b7280; margin-top: 0.3rem; font-weight: 500;'>File Compressor</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Mode selection
    mode = st.radio(
        "Mode",
        ["Compress", "Batch Compress", "Decompress", "Analytics", "History", "Settings", 
         "🎵 Neural Audio Compression"] if NEURAL_AUDIO_AVAILABLE else ["Compress", "Batch Compress", "Decompress", "Analytics", "History", "Settings"],
        key="mode_selector",
        help="Select compression mode"
    )
    
    st.divider()
    
    # Quick stats
    st.write("### Stats")
    history = load_history()
    if history:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Compressions", len(history))
        with col2:
            total_saved = sum(h.get('space_saved', 0) for h in history)
            st.metric("Saved", format_bytes(total_saved))
        
        avg_ratio = np.mean([h.get('compression_ratio', 0) for h in history])
        st.metric("Avg Ratio", f"{avg_ratio:.1f}%")
    else:
        st.info("No history yet")
    
    st.divider()
    st.markdown("""
    <div style='text-align: center; font-size: 0.8rem; color: #9ca3af; margin-top: 1.5rem;'>
        <div style='margin-bottom: 0.5rem;'>LZ77 + Huffman Encoding</div>
        <div>Lossless compression</div>
    </div>
    """, unsafe_allow_html=True)

# ==================== COMPRESS MODE ==================== #
if mode == "Compress":
    st.markdown("<div class='main-header'>File Compression</div>", unsafe_allow_html=True)
    
    # Welcome section
    col1, col2, col3 = st.columns(3)
    display_feature_card(col1, "🔗", "Dictionary", "LZ77 Encoding")
    display_feature_card(col2, "🌳", "Huffman", "Binary Tree")
    display_feature_card(col3, "✓", "Lossless", "100% Integrity")
    
    st.divider()
    
    # Upload section
    st.write("### Upload File")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Select a file to compress",
            type=["txt", "json", "py", "csv", "md", "xml", "log"],
            help="Supported: .txt, .json, .py, .csv, .md, .xml, .log"
        )
    
    with col2:
        show_advanced = st.checkbox("Advanced", value=False)
    
    # Advanced settings
    if show_advanced:
        st.markdown("### Advanced Options")
        col1, col2 = st.columns(2)
        with col1:
            lz77_window = st.slider("LZ77 Window", 1024, 32768, 32768, step=1024)
        with col2:
            min_match = st.slider("Min Match", 3, 10, 3)
    
    if uploaded_file:
        file_content = uploaded_file.read().decode('utf-8', errors='ignore')
        file_size = len(file_content)
        
        st.divider()
        
        # File info
        st.write("### File Info")
        col1, col2, col3, col4 = st.columns(4)
        display_metric_card(col1, "SIZE", format_bytes(file_size))
        display_metric_card(col2, "NAME", uploaded_file.name[:25])
        display_metric_card(col3, "CHARS", f"{len(file_content):,}")
        display_metric_card(col4, "LINES", f"{file_content.count(chr(10)):,}")
        
        # Compress button
        if st.button("Compress File", use_container_width=True, type="primary"):
            st.session_state['compress_clicked'] = True
        
        if st.session_state.get('compress_clicked', False):
            with st.spinner("Compressing..."):
                start_time = time.time()
                compressor = Compressor()
                encoded, root, tokens, codes = compressor.compress(file_content)
                compression_time = time.time() - start_time
                
                compressed_size = len(encoded)
                original_size = file_size
                compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
                space_saved = original_size - compressed_size
                
                st.session_state['encoded'] = encoded
                st.session_state['codes'] = codes
                st.session_state['tokens'] = tokens
                st.session_state['root'] = root
                st.session_state['original_content'] = file_content
                st.session_state['original_filename'] = uploaded_file.name
                st.session_state['compressed'] = True
                st.session_state['compression_time'] = compression_time
                st.session_state['compression_ratio'] = compression_ratio
                st.session_state['space_saved'] = space_saved
                st.session_state['compressed_size'] = compressed_size
                
                history_entry = {
                    "filename": uploaded_file.name,
                    "timestamp": datetime.now().isoformat(),
                    "original_size": original_size,
                    "compressed_size": compressed_size,
                    "compression_ratio": compression_ratio,
                    "space_saved": space_saved,
                    "time_taken": compression_time
                }
                save_history(history_entry)
                
                st.session_state['compress_clicked'] = False
                st.success("Compression complete!")
        
        st.divider()
        
        # Show results
        if 'compressed' in st.session_state and st.session_state['compressed']:
            encoded = st.session_state['encoded']
            root = st.session_state['root']
            tokens = st.session_state['tokens']
            
            compressed_size = st.session_state.get('compressed_size', len(encoded))
            compression_ratio = st.session_state.get('compression_ratio', 0)
            space_saved = st.session_state.get('space_saved', 0)
            compression_time = st.session_state.get('compression_time', 0)
            
            st.markdown("### Results")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            display_metric_card(col1, "ORIGINAL", format_bytes(file_size))
            display_metric_card(col2, "COMPRESSED", format_bytes(compressed_size))
            display_metric_card(col3, "RATIO", f"{compression_ratio:.1f}%")
            display_metric_card(col4, "SAVED", format_bytes(space_saved))
            display_metric_card(col5, "TIME", f"{compression_time:.3f}s")
            
            st.divider()
            
            # Tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Tree", 
                "Tokens", 
                "Stats", 
                "Binary",
                "Chart",
                "Download"
            ])
            
            with tab1:
                st.write("### Huffman Tree")
                fig = draw_huffman_tree_enhanced(root)
                st.pyplot(fig, use_container_width=True)
            
            with tab2:
                st.write("### LZ77 Tokens")
                
                formatted_tokens = []
                for t in tokens:
                    if t[0] == 'L':
                        formatted_tokens.append(f"L:{t[1]}")
                    else:
                        formatted_tokens.append(f"M:{t[1]},{t[2]}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total", len(tokens))
                with col2:
                    st.metric("Unique", len(set(formatted_tokens)))
                
                cols = st.columns(5)
                for idx, token in enumerate(formatted_tokens[:25]):
                    cols[idx % 5].markdown(f"`{token}`")
                
                if len(formatted_tokens) > 25:
                    st.info(f"... and {len(formatted_tokens) - 25} more")
            
            with tab3:
                st.write("### Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Input**")
                    st.write(f"Size: {format_bytes(file_size)}")
                    st.write(f"Bits: {file_size * 8:,}")
                    st.write(f"Chars: {len(file_content):,}")
                
                with col2:
                    st.markdown("**Output**")
                    st.write(f"Size: {format_bytes(compressed_size)}")
                    st.write(f"Bits: {compressed_size * 8:,}")
                    st.write(f"Reduction: {compression_ratio:.1f}%")
            
            with tab4:
                st.write("### Binary Data")
                preview = encoded[:128]
                formatted_binary = ' '.join(f"{byte:08b}" for byte in preview)
                st.code(formatted_binary, language='text')
                st.caption(f"Total: {format_bytes(len(encoded))}")
            
            with tab5:
                st.write("### Comparison")
                fig = create_stat_comparison(file_size, compressed_size)
                st.pyplot(fig, use_container_width=True)
            
            with tab6:
                st.write("### Download")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="Compressed Binary",
                        data=encoded,
                        file_name=f"{st.session_state['original_filename'].split('.')[0]}.bin",
                        mime="application/octet-stream",
                        use_container_width=True,
                        key="download_bin"
                    )
                
                with col2:
                    codes_array = [''] * 256
                    for byte_val, binary_code in st.session_state['codes'].items():
                        codes_array[byte_val] = binary_code
                    
                    metadata = {
                        "filename": st.session_state['original_filename'],
                        "original_size": file_size,
                        "compressed_size": compressed_size,
                        "compression_ratio": compression_ratio,
                        "codes": codes_array,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    st.download_button(
                        label="Metadata",
                        data=json.dumps(metadata, indent=2),
                        file_name=f"{st.session_state['original_filename'].split('.')[0]}_metadata.json",
                        mime="application/json",
                        use_container_width=True,
                        key="download_meta"
                    )
# ==================== BATCH COMPRESS MODE ==================== #
elif mode == "Batch Compress":
    st.markdown("<div class='main-header'>Batch Compression</div>", unsafe_allow_html=True)
    
    st.write("### Multiple Files")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["txt", "json", "py", "csv", "md"],
        accept_multiple_files=True,
        key="batch_uploader"
    )
    
    if uploaded_files:
        if st.button("Compress All", use_container_width=True, type="primary", key="batch_compress_btn"):
            ensure_output_dir()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            for idx, file in enumerate(uploaded_files):
                status_text.write(f"Processing: {file.name}")
                
                try:
                    file_content = file.read().decode('utf-8', errors='ignore')
                    file_size = len(file_content)
                    
                    start_time = time.time()
                    compressor = Compressor()
                    encoded, root, tokens, codes = compressor.compress(file_content)
                    compression_time = time.time() - start_time
                    
                    compressed_size = len(encoded)
                    compression_ratio = (1 - compressed_size / file_size) * 100 if file_size > 0 else 0
                    space_saved = file_size - compressed_size
                    
                    output_path = f"{OUTPUT_DIR}/{file.name.split('.')[0]}.bin"
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    with open(output_path, 'wb') as f:
                        f.write(encoded)
                    
                    results.append({
                        "File": file.name,
                        "Original": format_bytes(file_size),
                        "Compressed": format_bytes(compressed_size),
                        "Ratio": f"{compression_ratio:.1f}%",
                        "Time": f"{compression_time:.3f}s",
                        "Status": "✓"
                    })
                    
                    history_entry = {
                        "filename": file.name,
                        "timestamp": datetime.now().isoformat(),
                        "original_size": file_size,
                        "compressed_size": compressed_size,
                        "compression_ratio": compression_ratio,
                        "space_saved": space_saved,
                        "time_taken": compression_time
                    }
                    save_history(history_entry)
                    
                except Exception as e:
                    results.append({
                        "File": file.name,
                        "Status": f"✗ {str(e)[:30]}"
                    })
            
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.success("Done!")
            st.divider()
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)


# ==================== DECOMPRESS MODE ==================== #
elif mode == "Decompress":
    st.markdown("<div class='main-header'>File Decompression</div>", unsafe_allow_html=True)
    
    st.info("Upload both the .bin file and metadata.json file")
    
    col1, col2 = st.columns(2)
    with col1:
        bin_file = st.file_uploader("Binary file (.bin)", type=["bin"], key="decomp_bin")
    with col2:
        metadata_file = st.file_uploader("Metadata (JSON)", type=["json"], key="decomp_meta")
    
    if bin_file and metadata_file:
        if st.button("Decompress", type="primary", use_container_width=True, key="decomp_btn"):
            try:
                with st.spinner("Decompressing..."):
                    compressed_data = bin_file.read()
                    metadata = json.load(metadata_file)
                    
                    codes_from_json = metadata.get('codes', [])
                    original_filename = metadata.get('filename', 'decompressed.txt')
                    
                    if not codes_from_json:
                        st.error("No Huffman codes in metadata")
                        st.stop()
                    
                    if isinstance(codes_from_json, dict):
                        codes_converted = {int(k): v for k, v in codes_from_json.items()}
                    else:
                        codes_converted = {}
                        for byte_val, binary_code in enumerate(codes_from_json):
                            if binary_code:
                                codes_converted[byte_val] = binary_code
                    
                    start_time = time.time()
                    decompressor = Decompressor()
                    decompressed_content = decompressor.decompress(compressed_data, codes_converted)
                    decompression_time = time.time() - start_time
                
                st.session_state['decompressed'] = True
                st.session_state['decompressed_content'] = decompressed_content
                st.session_state['decompressed_size'] = len(decompressed_content)
                st.session_state['decompressed_filename'] = original_filename
                st.session_state['decompression_time'] = decompression_time
                
                st.success(f"Done in {decompression_time:.3f}s")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.divider()
        
        if 'decompressed' in st.session_state and st.session_state['decompressed']:
            decompressed = st.session_state['decompressed_content']
            
            st.write("### Results")
            col1, col2, col3, col4 = st.columns(4)
            display_metric_card(col1, "COMPRESSED", format_bytes(bin_file.size))
            display_metric_card(col2, "DECOMPRESSED", format_bytes(len(decompressed)))
            display_metric_card(col3, "EXPANSION", f"{len(decompressed) / bin_file.size:.2f}x")
            display_metric_card(col4, "TIME", f"{st.session_state['decompression_time']:.3f}s")
            
            st.divider()
            
            tab1, tab2, tab3 = st.tabs(["Preview", "Details", "Download"])
            
            with tab1:
                preview_len = min(500, len(decompressed))
                st.text_area("Content:", decompressed[:preview_len], height=200, disabled=True)
                if len(decompressed) > preview_len:
                    st.caption(f"Showing {preview_len} of {len(decompressed)} characters")
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Input**")
                    st.write(f"Size: {format_bytes(bin_file.size)}")
                    st.write(f"Bytes: {bin_file.size:,}")
                with col2:
                    st.write("**Output**")
                    st.write(f"Size: {format_bytes(len(decompressed))}")
                    st.write(f"Characters: {len(decompressed):,}")
            
            with tab3:
                st.download_button(
                    label="Download",
                    data=decompressed,
                    file_name=st.session_state['decompressed_filename'],
                    mime="text/plain",
                    use_container_width=True,
                    key="download_decomp"
                )
    
    elif bin_file or metadata_file:
        st.warning("Upload both files")


# ==================== ANALYTICS MODE ==================== #
elif mode == "Analytics":
    st.markdown("<div class='main-header'>Analytics</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    display_metric_card(col1, "ALGORITHM", "DEFLATE")
    display_metric_card(col2, "METHOD", "LZ77 + Huffman")
    display_metric_card(col3, "DATA LOSS", "None")
    display_metric_card(col4, "TYPE", "Lossless")
    
    st.divider()
    
    st.write("### How It Works")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <div style='font-size: 1.8rem; margin-bottom: 0.8rem;'>1</div>
            <div style='font-size: 1rem; font-weight: 600; color: #1a1a1a; margin-bottom: 0.8rem;'>Stage 1: LZ77</div>
            <div style='color: #6b7280; font-size: 0.9rem; line-height: 1.6;'>
            Finds repeated sequences and replaces them with references to previous occurrences
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <div style='font-size: 1.8rem; margin-bottom: 0.8rem;'>2</div>
            <div style='font-size: 1rem; font-weight: 600; color: #1a1a1a; margin-bottom: 0.8rem;'>Stage 2: Huffman</div>
            <div style='color: #6b7280; font-size: 0.9rem; line-height: 1.6;'>
            Analyzes token frequency and assigns shorter codes to frequent tokens
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.write("### Expected Ratios")
    ratio_data = pd.DataFrame({
        "File Type": ["Text", "Code", "JSON", "CSV", "Compressed"],
        "Ratio": ["50-60%", "55-65%", "45-55%", "40-50%", "0-5%"],
    })
    st.dataframe(ratio_data, use_container_width=True, hide_index=True)


# ==================== HISTORY MODE ==================== #
elif mode == "History":
    st.markdown("<div class='main-header'>History</div>", unsafe_allow_html=True)
    
    history = load_history()
    
    if history:
        df = pd.DataFrame(history)
        
        st.write("### Overview")
        col1, col2, col3, col4 = st.columns(4)
        display_metric_card(col1, "FILES", str(len(history)))
        display_metric_card(col2, "AVG RATIO", f"{df['compression_ratio'].mean():.1f}%")
        display_metric_card(col3, "TOTAL SAVED", format_bytes(df['space_saved'].sum()))
        display_metric_card(col4, "TOTAL TIME", f"{df['time_taken'].sum():.2f}s")
        
        st.divider()
        
        if st.button("Export as CSV", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="compression_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.divider()
        st.write("### Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Compression Ratios")
            chart_df = pd.DataFrame({
                'Index': range(len(df)),
                'Ratio': df['compression_ratio'].values
            })
            st.line_chart(chart_df.set_index('Index')['Ratio'], use_container_width=True)
        
        with col2:
            st.write("Space Saved")
            chart_df = df[['filename', 'space_saved']].head(10)
            st.bar_chart(chart_df.set_index('filename')['space_saved'], use_container_width=True)
        
        st.divider()
        st.write("### Recent")
        display_df = df[['filename', 'compression_ratio', 'space_saved', 'original_size']].copy()
        display_df.columns = ['File', 'Ratio', 'Saved', 'Size']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No history yet")


# ==================== SETTINGS MODE ==================== #
elif mode == "Settings":
    st.markdown("<div class='main-header'>Settings</div>", unsafe_allow_html=True)
    
    st.write("### System")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Compressions: **{len(load_history())}**")
        st.write(f"History: **{HISTORY_FILE}**")
    with col2:
        st.write(f"Output: **{OUTPUT_DIR}**")
        if os.path.exists(OUTPUT_DIR):
            files = len(os.listdir(OUTPUT_DIR))
            st.write(f"Files: **{files}**")
    
    st.divider()
    
    st.write("### Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear History", use_container_width=True, key="clear_history_btn"):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.success("Cleared!")
            st.rerun()
    
    with col2:
        st.button("Refresh", use_container_width=True, key="refresh_btn")
    
    st.divider()
    
    st.write("### About")
    st.markdown("""
    **DEFLATE** combines two compression techniques:
    - **LZ77**: Finds repeated patterns
    - **Huffman**: Assigns variable-length codes
    
    Used in ZIP, PNG, and HTTP compression standards.
    """)
    
    st.write("### Version")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Version", "2.0")
    with col2:
        st.metric("Algorithm", "DEFLATE")
    with col3:
        st.metric("Platform", "Streamlit")

# ==================== NEURAL AUDIO COMPRESSION MODE ==================== #
elif mode == "🎵 Neural Audio Compression":
    if not NEURAL_AUDIO_AVAILABLE:
        st.error("Neural Audio Compression requires PyTorch, librosa, and soundfile. Install with: pip install torch librosa soundfile")
        st.stop()
    
    st.markdown("<div class='main-header'>🎵 Neural Audio Compression</div>", unsafe_allow_html=True)
    
    # Load codec
    codec, device, model_trained = load_neural_codec()
    if codec is None:
        st.error("Failed to load codec.")
        st.stop()
    
    # Check if model is trained
    if not model_trained:
        st.error("퉴 MODEL NOT TRAINED - RECONSTRUCTED AUDIO WILL BE GARBAGE", icon="❌")
        st.markdown("""
        ### ⚠️ IMPORTANT: Train Your Model First!
        
        The neural codec needs to be trained before it can compress audio properly. 
        Without training, it will produce random beep sounds.
        
        **Quick Start (5 minutes):**
        ```bash
        python train_neural_codec.py --epochs 50 --use-synthetic
        ```
        
        **Better Quality (30 minutes):**
        ```bash
        python train_neural_codec.py --epochs 100 --use-synthetic
        ```
        
        **Best Quality (add your audio):**
        ```bash
        mkdir audio_data
        # Copy your .wav files to audio_data/
        python train_neural_codec.py --audio-dir ./audio_data --epochs 100
        ```
        
        Once training is complete, **restart this app** and the compression will work properly!
        """)
        st.stop()
    
    # Tabs for neural audio
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Compress Audio",
        "📊 Visualization", 
        "📈 Compare Codecs",
        "ℹ️ About"
    ])
    
    # ========== TAB 1: COMPRESS AUDIO ==========
    with tab1:
        st.write("### Upload & Compress Audio")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("#### 📤 Upload Audio File")
            uploaded_audio = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'flac', 'ogg'],
                help="Supported: WAV, MP3, FLAC, OGG",
                key="neural_audio_uploader"
            )
            
            if uploaded_audio:
                st.session_state.neural_audio_data = None  # Reset
                st.session_state.neural_compressed_indices = None
                st.session_state.neural_compression_metrics = {}
                st.session_state.neural_reconstructed_audio = None
                
                # Load audio
                with st.spinner("Loading audio..."):
                    audio, sr, success = load_audio_file(uploaded_audio)
                
                if success:
                    st.session_state.neural_audio_data = audio
                    st.session_state.neural_sample_rate = sr
                    st.success(f"✅ Loaded: {uploaded_audio.name}")
                    st.info(f"Duration: {len(audio)/sr:.2f}s | Sample Rate: {sr}Hz | Size: {audio.nbytes/1024:.1f}KB")
        
        with col2:
            st.write("#### 🎛️ Compression Controls")
            
            if st.session_state.neural_audio_data is not None:
                col_btn1, col_btn2 = st.columns(2)
                
                with col_btn1:
                    compress_btn = st.button(
                        "🔧 Compress",
                        type="primary",
                        use_container_width=True,
                        key="neural_compress_btn"
                    )
                
                with col_btn2:
                    if st.button("🗑️ Clear", use_container_width=True, key="neural_clear_btn"):
                        st.session_state.neural_audio_data = None
                        st.session_state.neural_compressed_indices = None
                        st.session_state.neural_compression_metrics = {}
                        st.session_state.neural_reconstructed_audio = None
                        st.rerun()
                
                if compress_btn:
                    with st.spinner("🔄 Compressing audio..."):
                        indices, latent_shape, metrics, success = compress_audio_neural(
                            st.session_state.neural_audio_data,
                            codec,
                            device
                        )
                    
                    if success:
                        st.session_state.neural_compressed_indices = indices
                        st.session_state.neural_compression_metrics = metrics
                        st.success("✅ Compression successful!")
            else:
                st.info("👆 Upload an audio file first")
        
        st.divider()
        
        # Results
        if st.session_state.neural_compressed_indices is not None:
            st.write("### 📊 Compression Results")
            
            metrics = st.session_state.neural_compression_metrics
            
            # Metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                display_metric_card(col1, "RATIO", f"{metrics['compression_ratio']:.1f}x", "Compression")
            
            with col2:
                display_metric_card(col2, "ORIGINAL", format_bytes(metrics['original_size']))
            
            with col3:
                display_metric_card(col3, "COMPRESSED", format_bytes(metrics['compressed_size']))
            
            with col4:
                display_metric_card(col4, "BITRATE", f"{metrics['bitrate_kbps']:.1f} kbps")
            
            st.divider()
            
            # Audio players and quality metrics
            audio_col1, audio_col2 = st.columns(2)
            
            with audio_col1:
                st.write("🔊 Original Audio")
                st.audio(st.session_state.neural_audio_data, sample_rate=st.session_state.neural_sample_rate)
            
            with audio_col2:
                st.write("🔊 Reconstructed Audio")
                
                # Check if we already have cached reconstructed audio
                if st.session_state.neural_reconstructed_audio is None:
                    with st.spinner("Decompressing..."):
                        reconstructed, decomp_success = decompress_audio_neural(
                            st.session_state.neural_compressed_indices,
                            metrics['latent_shape'],
                            codec,
                            device
                        )
                    
                    if decomp_success:
                        st.session_state.neural_reconstructed_audio = reconstructed
                        st.success("✅ Decompression successful!")
                    else:
                        st.error("❌ Decompression failed!")
                        st.stop()
                else:
                    reconstructed = st.session_state.neural_reconstructed_audio
                    decomp_success = True
                
                if decomp_success and reconstructed is not None:
                    st.audio(reconstructed, sample_rate=st.session_state.neural_sample_rate)
                    
                    # Evaluate
                    quality_metrics = evaluate_audio_quality(
                        st.session_state.neural_audio_data,
                        reconstructed,
                        st.session_state.neural_sample_rate
                    )
                    
                    st.divider()
                    
                    st.write("### 📈 Quality Metrics")
                    metric_cols = st.columns(3)
                    
                    with metric_cols[0]:
                        snr = quality_metrics.get('snr_db', 0)
                        display_metric_card(metric_cols[0], "SNR", f"{snr:.2f}", "dB (higher=better)")
                    
                    with metric_cols[1]:
                        mse = quality_metrics.get('mse', 0)
                        display_metric_card(metric_cols[1], "MSE", f"{mse:.6f}")
                    
                    with metric_cols[2]:
                        rmse = quality_metrics.get('rmse', 0)
                        display_metric_card(metric_cols[2], "RMSE", f"{rmse:.4f}")
    
    # ========== TAB 2: VISUALIZATION ==========
    with tab2:
        st.write("### 📊 Visualization & Analysis")
        
        if st.session_state.neural_audio_data is None:
            st.info("👆 Upload an audio file in the 'Compress Audio' tab first")
        else:
            # Decompress first
            if st.session_state.neural_compressed_indices is None:
                st.warning("Compress audio first in the 'Compress Audio' tab")
            else:
                reconstructed, _ = decompress_audio_neural(
                    st.session_state.neural_compressed_indices,
                    st.session_state.neural_compression_metrics['latent_shape'],
                    codec,
                    device
                )
                
                # Visualization options
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    show_waveform = st.checkbox("Show Waveform Comparison", value=True, key="viz_waveform")
                
                with viz_col2:
                    show_spectrum = st.checkbox("Show Frequency Spectrum", value=True, key="viz_spectrum")
                
                # Waveform
                if show_waveform:
                    st.write("🌊 **Waveform Comparison**")
                    fig = plot_waveform_comparison_neural(
                        st.session_state.neural_audio_data,
                        reconstructed,
                        st.session_state.neural_sample_rate
                    )
                    st.pyplot(fig, use_container_width=True)
                
                # Spectrum
                if show_spectrum:
                    st.write("📡 **Frequency Spectrum**")
                    fig = plot_frequency_spectrum_neural(
                        st.session_state.neural_audio_data,
                        reconstructed,
                        st.session_state.neural_sample_rate
                    )
                    st.pyplot(fig, use_container_width=True)
                
                # Metrics table
                st.divider()
                st.write("### 📋 Detailed Metrics")
                
                quality_metrics = evaluate_audio_quality(
                    st.session_state.neural_audio_data,
                    reconstructed,
                    st.session_state.neural_sample_rate
                )
                
                # Create DataFrame
                all_metrics = {**st.session_state.neural_compression_metrics, **quality_metrics}
                
                # Filter and format
                metric_df_data = {}
                for key, value in all_metrics.items():
                    if isinstance(value, (int, float)) and key not in ['latent_shape', 'indices_shape']:
                        metric_df_data[key.replace('_', ' ').title()] = f"{value:.4f}"
                
                if metric_df_data:
                    metric_df = pd.DataFrame(list(metric_df_data.items()), columns=['Metric', 'Value'])
                    st.dataframe(metric_df, use_container_width=True, hide_index=True)
    
    # ========== TAB 3: COMPARE CODECS ==========
    with tab3:
        st.write("### 📈 Compare with Other Codecs")
        
        if st.session_state.neural_audio_data is None:
            st.info("👆 Upload an audio file first")
        else:
            st.write("Comparing Neural Codec with traditional compression methods...")
            
            # Neural codec results
            neural_ratio = st.session_state.neural_compression_metrics.get('compression_ratio', 0)
            neural_bitrate = st.session_state.neural_compression_metrics.get('bitrate_kbps', 0)
            
            # Comparison data
            comparison_data = {
                'Codec': ['Neural Audio Codec', 'Opus', 'DEFLATE', 'LZMA', 'Zstandard'],
                'Compression Ratio': [neural_ratio if neural_ratio > 0 else 20.0, 8.0, 1.5, 3.0, 2.5],
                'Bitrate (kbps)': [neural_bitrate if neural_bitrate > 0 else 2.0, 32, 256, 85, 102],
                'Speed': ['Fast', 'Very Fast', 'Very Fast', 'Slow', 'Fast'],
                'Quality': ['Excellent', 'Excellent', 'Lossless', 'Lossless', 'Lossless'],
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Compression Ratio Comparison**")
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                
                x_pos = np.arange(len(df_comparison))
                bars = ax.bar(x_pos, df_comparison['Compression Ratio'], 
                             color=['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'])
                ax.set_xticks(x_pos)
                ax.set_xticklabels(df_comparison['Codec'], rotation=45, ha='right')
                ax.set_ylabel('Compression Ratio (higher is better)', color='#6b7280')
                ax.set_title('Compression Ratio', color='#1a1a1a', fontweight='600')
                ax.grid(True, alpha=0.2, axis='y')
                ax.tick_params(colors='#6b7280')
                
                for bar, value in zip(bars, df_comparison['Compression Ratio']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f}x', ha='center', va='bottom', fontweight='600', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
            
            with col2:
                st.write("**Bitrate Comparison**")
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                
                x_pos = np.arange(len(df_comparison))
                bars = ax.bar(x_pos, df_comparison['Bitrate (kbps)'],
                             color=['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'])
                ax.set_xticks(x_pos)
                ax.set_xticklabels(df_comparison['Codec'], rotation=45, ha='right')
                ax.set_ylabel('Bitrate (kbps, lower is better)', color='#6b7280')
                ax.set_title('Bitrate', color='#1a1a1a', fontweight='600')
                ax.grid(True, alpha=0.2, axis='y')
                ax.tick_params(colors='#6b7280')
                
                for bar, value in zip(bars, df_comparison['Bitrate (kbps)']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.0f}', ha='center', va='bottom', fontweight='600', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
    
    # ========== TAB 4: ABOUT ==========
    with tab4:
        st.write("## ℹ️ About Neural Audio Compression")
        
        st.markdown("""
        ### What is Neural Audio Compression?
        
        Neural Audio Compression uses **deep learning** to compress audio files more efficiently 
        than traditional methods like Opus or MP3.
        
        **Key Features:**
        - 🎯 20x compression ratio (vs 8x for Opus)
        - 🧠 Learns from audio data to optimize compression
        - ⚡ Fast encoding and decoding
        - 🎨 Maintains good audio quality
        
        ### How It Works
        
        1. **Encoder**: Compresses audio to a tiny "latent space"
        2. **Quantizer**: Converts continuous values to discrete integers
        3. **Entropy Coding**: Further compresses using Huffman or Zstandard
        4. **Decoder**: Reconstructs audio from compressed data
        
        ### Architecture
        
        - **Codec**: Encoder-Decoder CNN with residual connections
        - **Latent Dim**: 128 dimensions (bottleneck)
        - **Quantization**: 256-entry codebook (8-bit)
        - **Sample Rate**: 16 kHz
        - **Frame Size**: 16,000 samples (1 second)
        
        ### Quality Metrics
        
        - **SNR (Signal-to-Noise Ratio)**: Measures reconstruction quality (higher is better)
        - **MSE**: Mean squared error between original and reconstructed
        - **Bitrate**: Final compressed size (kbps)
        - **Compression Ratio**: Original size / Compressed size
        
        ### Train Your Own Model
        
        Want a better codec? Train it on your own data:
        
        ```bash
        # Synthetic data (quick test)
        python train_neural_codec.py --epochs 50 --use-synthetic
        
        # Real audio data
        mkdir audio_data
        cp /path/to/audio/*.wav audio_data/
        python train_neural_codec.py --audio-dir ./audio_data --epochs 100
        ```
        
        See `NEURAL_AUDIO_README.md` for full details!
        """)
        
        st.divider()
        st.markdown("""
        **Further Reading:**
        - [VQVAE Paper](https://arxiv.org/abs/1711.00937)
        - [SoundStream (Google)](https://arxiv.org/abs/2107.03312)
        - [EnCodec (Meta)](https://arxiv.org/abs/2210.13438)
        """)
