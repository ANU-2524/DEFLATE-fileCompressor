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
        ["Compress", "Batch Compress", "Decompress", "Analytics", "History", "Settings"],
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
if "Compress" in mode:
    st.markdown("<div class='main-header'>🗜️ File Compression Engine</div>", unsafe_allow_html=True)
    
    # Welcome section
    col1, col2, col3 = st.columns(3)
    display_feature_card(col1, "⚡", "LZ77", "Dictionary\nEncoding")
    display_feature_card(col2, "🌳", "Huffman", "Statistical\nEncoding")
    display_feature_card(col3, "🔒", "Lossless", "100% Data\nIntegrity")
    
    st.divider()
    
    # Upload section
    st.write("### 📤 Upload File")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Select a file to compress",
            type=["txt", "json", "py", "csv", "md", "xml", "log"],
            help="Supported: .txt, .json, .py, .csv, .md, .xml, .log"
        )
    
    with col2:
        show_advanced = st.checkbox("⚙️ Advanced", value=False)
    
    # Advanced settings
    if show_advanced:
        st.markdown("### 🔧 Advanced Options")
        col1, col2 = st.columns(2)
        with col1:
            lz77_window = st.slider("LZ77 Window", 1024, 32768, 32768, step=1024)
        with col2:
            min_match = st.slider("Min Match Length", 3, 10, 3)
    
    if uploaded_file:
        file_content = uploaded_file.read().decode('utf-8', errors='ignore')
        file_size = len(file_content)
        
        st.divider()
        
        # File info cards
        st.write("### 📊 File Information")
        col1, col2, col3, col4 = st.columns(4)
        display_metric_card(col1, "FILE SIZE", format_bytes(file_size), "Original")
        display_metric_card(col2, "FILE NAME", uploaded_file.name[:20])
        display_metric_card(col3, "CHARACTERS", f"{len(file_content):,}")
        display_metric_card(col4, "LINES", f"{file_content.count(chr(10)):,}")
        
        # Compress action
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("▶ Start Compression", use_container_width=True, type="primary"):
                st.session_state['compress_clicked'] = True
        
        if st.session_state.get('compress_clicked', False):
            with st.spinner("🔄 Compressing your file..."):
                progress_container = st.container()
                
                start_time = time.time()
                compressor = Compressor()
                encoded, root, tokens, codes = compressor.compress(file_content)
                compression_time = time.time() - start_time
                
                compressed_size = len(encoded)
                original_size = file_size
                compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
                space_saved = original_size - compressed_size
                
                # Store in session
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
                
                # Save to history
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
                st.success("✅ Compression Complete!")
        
        st.divider()
        
        # Show results if compressed
        if 'compressed' in st.session_state and st.session_state['compressed']:
            encoded = st.session_state['encoded']
            root = st.session_state['root']
            tokens = st.session_state['tokens']
            
            compressed_size = st.session_state.get('compressed_size', len(encoded))
            compression_ratio = st.session_state.get('compression_ratio', 0)
            space_saved = st.session_state.get('space_saved', 0)
            compression_time = st.session_state.get('compression_time', 0)
            
            # ========== COMPRESSION STATISTICS ========== #
            st.markdown("### 📈 Compression Results")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            display_metric_card(col1, "ORIGINAL", format_bytes(file_size))
            display_metric_card(col2, "COMPRESSED", format_bytes(compressed_size))
            display_metric_card(col3, "REDUCTION", f"{compression_ratio:.2f}%", delta="Smaller ✓")
            display_metric_card(col4, "SPACE SAVED", format_bytes(space_saved))
            display_metric_card(col5, "TIME TAKEN", f"{compression_time:.3f}s")
            
            st.divider()
            
            # ========== TABS ========== #
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🌳 Huffman Tree", 
                "🔗 LZ77 Tokens", 
                "📊 Statistics", 
                "💾 Binary Preview",
                "📉 Comparison",
                "⬇️ Download"
            ])
            
            # TAB 1: HUFFMAN TREE
            with tab1:
                st.write("### Huffman Encoding Tree Visualization")
                st.write("**Nodes:** Character or Frequency | **Edges:** Bit Codes (0/1)")
                fig = draw_huffman_tree_enhanced(root)
                st.pyplot(fig, use_container_width=True)
            
            # TAB 2: LZ77 TOKENS
            with tab2:
                st.write("### LZ77 Token Stream")
                st.write("**L:** Literal | **M:** Match (distance, length)")
                
                formatted_tokens = []
                for t in tokens:
                    if t[0] == 'L':
                        formatted_tokens.append(f"L:{t[1]}")
                    else:
                        formatted_tokens.append(f"M:{t[1]},{t[2]}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Tokens", len(tokens))
                with col2:
                    st.metric("Unique Tokens", len(set(formatted_tokens)))
                
                # Display tokens in a grid
                st.write("**Token Preview:**")
                cols = st.columns(5)
                for idx, token in enumerate(formatted_tokens[:25]):
                    cols[idx % 5].markdown(f"`{token}`")
                
                if len(formatted_tokens) > 25:
                    st.info(f"... and {len(formatted_tokens) - 25} more tokens")
            
            # TAB 3: STATISTICS
            with tab3:
                st.write("### Detailed Compression Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📥 Input Analysis**")
                    st.write(f"- File Size: **{format_bytes(file_size)}**")
                    st.write(f"- Bits: **{file_size * 8:,}**")
                    st.write(f"- Characters: **{len(file_content):,}**")
                    st.write(f"- Entropy Estimate: **{min(8, 8 * (1 - compression_ratio/100)):.2f} bits/byte**")
                
                with col2:
                    st.markdown("**📤 Output Analysis**")
                    st.write(f"- Compressed Size: **{format_bytes(compressed_size)}**")
                    st.write(f"- Bits: **{compressed_size * 8:,}**")
                    st.write(f"- Reduction: **{compression_ratio:.2f}%**")
                    st.write(f"- Effective Rate: **{(compressed_size * 8 / len(file_content)):.2f} bits/byte**")
                
                st.write("---")
                st.write("**🔗 Token Statistics**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Generated", len(tokens))
                with col2:
                    st.metric("Unique", len(set(formatted_tokens)))
                with col3:
                    st.metric("Per Char", f"{len(tokens)/len(file_content):.3f}")
            
            # TAB 4: BINARY PREVIEW
            with tab4:
                st.write("### Binary Output Preview")
                st.write(f"*Showing first 128 bytes of {format_bytes(len(encoded))} total*")
                preview = encoded[:128]
                formatted_binary = ' '.join(f"{byte:08b}" for byte in preview)
                
                with st.expander("View Binary Data", expanded=True):
                    st.code(formatted_binary, language='text')
                
                st.info(f"**Total Size:** {format_bytes(len(encoded))} ({len(encoded) * 8:,} bits)")
            
            # TAB 5: COMPARISON CHART
            with tab5:
                st.write("### Visual Comparison")
                
                fig = create_stat_comparison(file_size, compressed_size)
                st.pyplot(fig, use_container_width=True)
                
                # Detailed comparison table
                st.write("**Detailed Metrics:**")
                comparison_data = pd.DataFrame({
                    "Metric": ["File Size", "Data Bits", "Compression Ratio", "Time Required"],
                    "Original": [format_bytes(file_size), f"{file_size * 8:,}", "-", "-"],
                    "Compressed": [format_bytes(compressed_size), f"{compressed_size * 8:,}", f"{compression_ratio:.2f}%", f"{compression_time:.3f}s"],
                    "Difference": [f"-{format_bytes(space_saved)}", f"-{space_saved * 8:,}", f"{compression_ratio:.2f}%", "-"]
                })
                
                st.dataframe(comparison_data, use_container_width=True, hide_index=True)
            
            # TAB 6: DOWNLOAD
            with tab6:
                st.write("### 📥 Download Compressed Files")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📦 Compressed Binary (.bin)",
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
                        label="📋 Metadata (JSON)",
                        data=json.dumps(metadata, indent=2),
                        file_name=f"{st.session_state['original_filename'].split('.')[0]}_metadata.json",
                        mime="application/json",
                        use_container_width=True,
                        key="download_meta"
                    )
                
                st.info("💡 **Tip:** Keep both files together to decompress later!")
# ==================== BATCH COMPRESS MODE ==================== #
elif "Batch" in mode:
    st.markdown("<div class='main-header'>📦 Batch Compression</div>", unsafe_allow_html=True)
    
    st.write("### 📤 Select Multiple Files")
    uploaded_files = st.file_uploader(
        "Choose multiple files to compress",
        type=["txt", "json", "py", "csv", "md"],
        accept_multiple_files=True,
        key="batch_uploader"
    )
    
    if uploaded_files:
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("▶ Compress All Files", use_container_width=True, type="primary", key="batch_compress_btn"):
                ensure_output_dir()
                progress_container = st.container()
                results = []
                
                progress_bar = progress_container.progress(0)
                status_text = progress_container.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status_text.write(f"⏳ Processing: {file.name}")
                    
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
                        
                        # Save compressed file
                        output_path = f"{OUTPUT_DIR}/{file.name.split('.')[0]}.bin"
                        os.makedirs(OUTPUT_DIR, exist_ok=True)
                        with open(output_path, 'wb') as f:
                            f.write(encoded)
                        
                        results.append({
                            "📁 File": file.name,
                            "📥 Original": format_bytes(file_size),
                            "📤 Compressed": format_bytes(compressed_size),
                            "📊 Ratio": f"{compression_ratio:.2f}%",
                            "⏱️ Time": f"{compression_time:.3f}s",
                            "✅ Status": "Success"
                        })
                        
                        # Save to history
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
                            "📁 File": file.name,
                            "✅ Status": f"❌ Error: {str(e)[:30]}"
                        })
                
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                
                status_text.success("✅ Batch compression complete!")
                
                st.divider()
                st.write("### 📋 Results")
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                
                # Summary stats
                col1, col2, col3, col4 = st.columns(4)
                successful = len([r for r in results if "Success" in str(r.get("✅ Status", ""))])
                with col1:
                    st.metric("Total Files", len(uploaded_files))
                with col2:
                    st.metric("Successful", successful)
                with col3:
                    st.metric("Failed", len(uploaded_files) - successful)
                with col4:
                    st.metric("Output Dir", OUTPUT_DIR)
    else:
        st.info("👆 Upload files to get started")


# ==================== DECOMPRESS MODE ==================== #
elif "Decompres" in mode:
    st.markdown("<div class='main-header'>📂 File Decompression</div>", unsafe_allow_html=True)
    
    st.info("ℹ️ To decompress a file, you need:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("1️⃣ **Compressed `.bin` file**")
    with col2:
        st.markdown("2️⃣ **Metadata `.json` file**")
    with col3:
        st.markdown("3️⃣ **Both must match**")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 📦 Compressed Binary")
        bin_file = st.file_uploader("Upload .bin file", type=["bin"], key="decomp_bin")
    
    with col2:
        st.write("### 📋 Metadata File")
        metadata_file = st.file_uploader("Upload metadata.json", type=["json"], key="decomp_meta")
    
    if bin_file and metadata_file:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("▶ Decompress", type="primary", use_container_width=True, key="decomp_btn"):
                st.session_state['decompress_clicked'] = True
        
        if st.session_state.get('decompress_clicked', False):
            try:
                # Read files
                compressed_data = bin_file.read()
                metadata = json.load(metadata_file)
                
                with st.spinner("🔄 Decompressing..."):
                    codes_from_json = metadata.get('codes', [])
                    original_filename = metadata.get('filename', 'decompressed.txt')
                    original_size_meta = metadata.get('original_size', 0)
                    
                    if not codes_from_json:
                        st.error("❌ No Huffman codes in metadata!")
                        st.stop()
                    
                    # Decode codes
                    if isinstance(codes_from_json, dict):
                        codes_converted = {int(k): v for k, v in codes_from_json.items()}
                    else:
                        codes_converted = {}
                        for byte_val, binary_code in enumerate(codes_from_json):
                            if binary_code:
                                codes_converted[byte_val] = binary_code
                    
                    if not codes_converted:
                        st.error("❌ No valid Huffman codes found!")
                        st.stop()
                    
                    start_time = time.time()
                    decompressor = Decompressor()
                    decompressed_content = decompressor.decompress(compressed_data, codes_converted)
                    decompression_time = time.time() - start_time
                
                st.session_state['decompressed'] = True
                st.session_state['decompressed_content'] = decompressed_content
                st.session_state['decompressed_size'] = len(decompressed_content)
                st.session_state['decompressed_filename'] = original_filename
                st.session_state['decompression_time'] = decompression_time
                st.session_state['decompress_clicked'] = False
                
                st.success(f"✅ Decompression Complete! ({decompression_time:.3f}s)")
            
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON metadata file!")
            except Exception as e:
                st.error(f"❌ Decompression Error: {str(e)}")
        
        st.divider()
        
        # Show results
        if 'decompressed' in st.session_state and st.session_state['decompressed']:
            decompressed = st.session_state['decompressed_content']
            
            st.markdown("### 📈 Decompression Results")
            col1, col2, col3, col4 = st.columns(4)
            display_metric_card(col1, "COMPRESSED", format_bytes(len(compressed_data)))
            display_metric_card(col2, "DECOMPRESSED", format_bytes(len(decompressed)))
            display_metric_card(col3, "EXPANSION", f"{len(decompressed) / len(compressed_data):.2f}x")
            display_metric_card(col4, "TIME", f"{st.session_state['decompression_time']:.3f}s")
            
            st.divider()
            
            # Tabs
            tab1, tab2, tab3 = st.tabs(["👁️ Preview", "📊 Statistics", "⬇️ Download"])
            
            with tab1:
                st.write("### File Preview")
                preview_len = min(1000, len(decompressed))
                preview_text = decompressed[:preview_len]
                st.text_area("Content:", preview_text, height=250, disabled=True)
                if len(decompressed) > preview_len:
                    st.info(f"📄 Showing {preview_len} of {len(decompressed)} characters")
            
            with tab2:
                st.write("### Decompression Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Input (Compressed)**")
                    st.write(f"📦 Size: {format_bytes(len(compressed_data))}")
                    st.write(f"💾 Bytes: {len(compressed_data):,}")
                with col2:
                    st.markdown("**Output (Original)**")
                    st.write(f"📄 Size: {format_bytes(len(decompressed))}")
                    st.write(f"📝 Chars: {len(decompressed):,}")
                
                st.write("---")
                st.markdown("**🔑 Huffman Information**")
                st.write(f"✅ Codes Loaded: Yes")
                st.write(f"⏱️ Decompression Time: {st.session_state['decompression_time']:.4f}s")
            
            with tab3:
                st.write("### Download Decompressed File")
                st.download_button(
                    label="⬇️ Download",
                    data=decompressed,
                    file_name=st.session_state['decompressed_filename'],
                    mime="text/plain",
                    use_container_width=True,
                    key="download_decomp"
                )
                st.success(f"File will be saved as: **{st.session_state['decompressed_filename']}**")
    
    elif bin_file or metadata_file:
        st.warning("⚠️ Please upload BOTH .bin and metadata.json files")


# ==================== ANALYTICS MODE ==================== #
elif "Analytics" in mode:
    st.markdown("<div class='main-header'>📊 Compression Analytics</div>", unsafe_allow_html=True)
    
    # Algorithm info cards
    col1, col2, col3, col4 = st.columns(4)
    display_metric_card(col1, "ALGORITHM", "DEFLATE")
    display_metric_card(col2, "COMPONENTS", "LZ77 + Huffman")
    display_metric_card(col3, "DATA LOSS", "None (Lossless)")
    display_metric_card(col4, "USE CASES", "ZIP, PNG, HTTP")
    
    st.divider()
    
    st.markdown("### 🔄 How DEFLATE Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🔗</div>
            <div style='font-size: 1.2rem; font-weight: 700; color: #e2e8f0; margin-bottom: 1rem;'>Stage 1: LZ77</div>
            <div style='color: #94a3b8;'>
            • Finds repeated sequences<br>
            • Replaces with references<br>
            • Creates token stream<br>
            • Typical: 20-30% reduction
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🌳</div>
            <div style='font-size: 1.2rem; font-weight: 700; color: #e2e8f0; margin-bottom: 1rem;'>Stage 2: Huffman</div>
            <div style='color: #94a3b8;'>
            • Analyzes token frequency<br>
            • Builds binary tree<br>
            • Assigns variable-length codes<br>
            • Typical: Additional 30-40% reduction
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 📈 Expected Compression Ratios")
    ratio_data = pd.DataFrame({
        "📁 File Type": ["Text Files", "Source Code", "JSON/XML", "CSV Data", "Already Compressed"],
        "📊 Ratio": ["50-60%", "55-65%", "45-55%", "40-50%", "0-5%"],
        "💡 Best For": ["Books, Logs", "Python, JS", "APIs, Config", "Datasets", "Images, Video"]
    })
    st.dataframe(ratio_data, use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.markdown("### ⚡ Performance Tips")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **✅ Good for compression:**
        - Text files
        - Source code
        - Log files
        - Configuration files
        - JSON/CSV data
        """)
    with col2:
        st.markdown("""
        **❌ Not ideal:**
        - Already compressed
        - Binary data
        - Images (PNG, JPG)
        - Audio/Video
        - Encrypted data
        """)


# ==================== HISTORY MODE ==================== #
elif "History" in mode:
    st.markdown("<div class='main-header'>📜 Compression History</div>", unsafe_allow_html=True)
    
    history = load_history()
    
    if history:
        st.divider()
        
        # Overview statistics
        df = pd.DataFrame(history)
        
        col1, col2, col3, col4 = st.columns(4)
        display_metric_card(col1, "TOTAL FILES", str(len(history)))
        display_metric_card(col2, "AVG RATIO", f"{df['compression_ratio'].mean():.2f}%")
        display_metric_card(col3, "TOTAL SAVED", format_bytes(df['space_saved'].sum()))
        display_metric_card(col4, "TOTAL TIME", f"{df['time_taken'].sum():.2f}s")
        
        st.divider()
        
        # Export option
        if st.button("📥 Export as CSV", use_container_width=True, key="export_csv"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="compression_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.divider()
        
        # Charts
        st.markdown("### 📊 Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Compression Ratios Over Time")
            chart_df = pd.DataFrame({
                'Index': range(len(df)),
                'Ratio': df['compression_ratio'].values
            })
            st.line_chart(chart_df.set_index('Index')['Ratio'], use_container_width=True)
        
        with col2:
            st.write("### Space Saved by File")
            chart_df = df[['filename', 'space_saved']].head(10)
            st.bar_chart(chart_df.set_index('filename')['space_saved'], use_container_width=True)
        
        st.divider()
        
        # Detailed history table
        st.write("### 📋 Recent Compressions")
        display_df = df[['filename', 'compression_ratio', 'space_saved', 'original_size', 'compressed_size']].copy()
        display_df['compression_ratio'] = display_df['compression_ratio'].apply(lambda x: f"{x:.2f}%")
        display_df['space_saved'] = display_df['space_saved'].apply(format_bytes)
        display_df['original_size'] = display_df['original_size'].apply(format_bytes)
        display_df['compressed_size'] = display_df['compressed_size'].apply(format_bytes)
        display_df.columns = ['📁 File', '📊 Ratio', '💾 Saved', '📥 Original', '📤 Compressed']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    else:
        st.info("📭 No compression history yet. Start compressing to see it here!")


# ==================== SETTINGS MODE ==================== #
elif "Settings" in mode:
    st.markdown("<div class='main-header'>⚙️ Settings</div>", unsafe_allow_html=True)
    
    # System info
    st.markdown("### 🖥️ System Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"📊 Total Compressions: **{len(load_history())}**")
        st.write(f"💾 History File: **{HISTORY_FILE}**")
    with col2:
        st.write(f"📁 Output Directory: **{OUTPUT_DIR}**")
        st.write(f"⏰ Current Time: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")
    
    st.divider()
    
    # History management
    st.markdown("### 📁 History Management")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear All History", use_container_width=True, key="clear_history_btn"):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.success("✅ History cleared!")
            st.rerun()
    
    with col2:
        if st.button("📂 Open Output Folder", use_container_width=True, key="open_output"):
            st.info(f"Output folder: {os.path.abspath(OUTPUT_DIR)}")
    
    with col3:
        if os.path.exists(OUTPUT_DIR):
            files = os.listdir(OUTPUT_DIR)
            st.metric("Compressed Files", len(files))
    
    st.divider()
    
    # About section
    st.markdown("### 📖 About DEFLATE")
    st.markdown("""
    **DEFLATE** is a lossless data compression algorithm that combines two powerful techniques:
    
    **🔗 LZ77 Dictionary Encoding**
    - Finds repeated patterns in data
    - Replaces duplicates with positional references
    - Typically achieves 20-30% reduction
    
    **🌳 Huffman Statistical Encoding**
    - Analyzes frequency of tokens
    - Builds a binary tree structure
    - Assigns shorter codes to frequent items
    - Provides additional 30-40% reduction
    
    **Real-world Usage:**
    - ZIP archives
    - PNG images
    - HTTP compression (DEFLATE)
    - gz compression
    """)
    
    st.divider()
    
    st.markdown("### 📊 Version & Credits")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("App Version", "Pro 2.0")
    with col2:
        st.metric("Algorithm", "LZ77 + Huffman")
    with col3:
        st.metric("Platform", "Streamlit")
