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

# ==================== PAGE CONFIG ==================== #
st.set_page_config(
    page_title="DEFLATE Compressor Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE INITIALIZATION ==================== #
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

if 'mode' not in st.session_state:
    st.session_state.mode = 'Compress'

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if 'compressed_data' not in st.session_state:
    st.session_state.compressed_data = None

# ==================== CUSTOM STYLING ==================== #
if st.session_state.theme == 'dark':
    bg_color = "#0e1117"
    text_color = "#ffffff"
    card_color = "#161b22"
else:
    bg_color = "#ffffff"
    text_color = "#1f1f1f"
    card_color = "#f0f2f6"

st.markdown(f"""
    <style>
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }}
    .subheader {{
        font-size: 1.8rem;
        font-weight: bold;
        color: #2e5090;
        margin: 1.5rem 0 1rem 0;
    }}
    .metric-card {{
        background-color: {card_color};
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }}
    .stat-title {{
        font-size: 0.85rem;
        font-weight: 600;
        opacity: 0.7;
    }}
    .stat-value {{
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
    }}
    .history-item {{
        background-color: {card_color};
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #17a2b8;
    }}
    </style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS ==================== #
HISTORY_FILE = "compression_history.json"
OUTPUT_DIR = "output"

# ==================== HELPER FUNCTIONS ==================== #
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
    
    # Keep only last 50 entries
    history = history[-50:]
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

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
    """Draw Huffman tree with graphviz layout"""
    G = nx.DiGraph()

    def add_edges(node, parent=None):
        if node:
            # Handle both char attribute and integer keys
            if hasattr(node, 'char'):
                if node.char is not None:
                    # If it's an integer (byte value), show as chr representation
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

    # Try graphviz layout first
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except:
        pos = hierarchy_pos(G, id(root))

    labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')

    fig, ax = plt.subplots(figsize=(14, 8))
    nx.draw(G, pos, labels=labels, with_labels=True, ax=ax,
            node_color='lightblue', node_size=1200,
            font_size=9, font_weight='bold',
            arrows=True, arrowsize=15, edge_color='gray', width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)
    
    ax.set_title("Huffman Tree (Hierarchical Visualization)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def ensure_output_dir():
    """Ensure output directory exists"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def export_report(history_data):
    """Create a report from compression history"""
    if not history_data:
        return None
    
    df = pd.DataFrame(history_data)
    return df

# ==================== SIDEBAR ==================== #
st.sidebar.title("DEFLATE Compressor Pro")
st.sidebar.divider()

# Theme toggle
col_theme1, col_theme2 = st.sidebar.columns(2)
with col_theme1:
    if st.button("🌙 Dark" if st.session_state.theme == 'light' else "☀️ Light"):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()

st.sidebar.write(f"Theme: {st.session_state.theme.upper()}")
st.sidebar.divider()

# Mode selection
mode = st.sidebar.radio(
    "Select Mode:",
    ["Compress", "Batch Compress", "Decompress", "Analytics", "History", "Settings"],
    key="mode_selector",
    help="Choose compression mode"
)

st.sidebar.divider()
st.sidebar.write("### Quick Stats")
history = load_history()
if history:
    st.sidebar.metric("Total Compressions", len(history))
    total_saved = sum(h.get('space_saved', 0) for h in history)
    st.sidebar.metric("Total Space Saved", format_bytes(total_saved))

# ==================== COMPRESS MODE ==================== #
if mode == "Compress":
    st.markdown("<div class='main-header'>File Compression Engine</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload a text file to compress",
            type=["txt", "json", "py", "csv", "md", "xml"],
            help="Supported formats: .txt, .json, .py, .csv, .md, .xml"
        )
    
    with col2:
        st.write("")
        show_advanced = st.checkbox("Advanced Settings", value=False)
    
    # Advanced settings
    if show_advanced:
        st.write("### Advanced Options")
        lz77_window = st.slider("LZ77 Window Size", 1024, 32768, 32768, step=1024)
        min_match = st.slider("Min Match Length", 3, 10, 3)
    
    if uploaded_file:
        file_content = uploaded_file.read().decode('utf-8', errors='ignore')
        file_size = len(file_content)
        
        st.divider()
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Size", format_bytes(file_size), "Original")
        with col2:
            st.metric("File Name", uploaded_file.name)
        with col3:
            st.metric("Characters", f"{len(file_content):,}")
        
        # Compress button
        if st.button("▶ Start Compression", use_container_width=True, type="primary", key="compress_btn"):
            with st.spinner("Compressing..."):
                start_time = time.time()
                compressor = Compressor()
                encoded, root, tokens, codes = compressor.compress(file_content)
                compression_time = time.time() - start_time
                
                # encoded is now bytes (properly packed binary)
                compressed_size = len(encoded)  # size in bytes
                original_size = file_size  # size in bytes
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
                
                st.success("✅ Compression Complete!")
        
        st.divider()
        
        # Show results if compressed
        if 'compressed' in st.session_state and st.session_state['compressed']:
            encoded = st.session_state['encoded']
            root = st.session_state['root']
            tokens = st.session_state['tokens']
            
            compressed_size = st.session_state.get('compressed_size', len(encoded))  # in bytes
            compression_ratio = st.session_state.get('compression_ratio', 0)
            space_saved = st.session_state.get('space_saved', 0)
            compression_time = st.session_state.get('compression_time', 0)
            
            # ========== COMPRESSION STATISTICS ========== #
            st.subheader("Compression Results")
            
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
            
            with metric_col1:
                st.metric("Original", format_bytes(file_size))
            with metric_col2:
                st.metric("Compressed", format_bytes(compressed_size))
            with metric_col3:
                st.metric("Ratio", f"{compression_ratio:.2f}%")
            with metric_col4:
                st.metric("Saved", format_bytes(space_saved))
            with metric_col5:
                st.metric("Time", f"{compression_time:.3f}s")
            
            st.divider()
            
            # ========== TABS ========== #
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Huffman Tree", 
                "LZ77 Tokens", 
                "Statistics", 
                "Binary Preview",
                "Comparison",
                "Download"
            ])
            
            # TAB 1: HUFFMAN TREE
            with tab1:
                st.write("### Huffman Encoding Tree")
                st.write("Nodes: Character or Frequency | Edges: Bit Code (0/1)")
                fig = draw_huffman_tree_enhanced(root)
                st.pyplot(fig)
            
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
                
                # Display in columns
                cols = st.columns(4)
                for idx, token in enumerate(formatted_tokens[:20]):  # Show first 20
                    cols[idx % 4].write(f"`{token}`")
                
                if len(formatted_tokens) > 20:
                    st.info(f"... and {len(formatted_tokens) - 20} more tokens")
                
                st.metric("Total Tokens", len(tokens))
            
            # TAB 3: STATISTICS
            with tab3:
                st.write("### Detailed Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Input Analysis**")
                    st.write(f"- Size: {format_bytes(file_size)}")
                    st.write(f"- Bits: {file_size * 8:,}")
                    st.write(f"- Characters: {len(file_content):,}")
                
                with col2:
                    st.write("**Output Analysis**")
                    st.write(f"- Size: {format_bytes(compressed_size)}")
                    st.write(f"- Bits: {compressed_size * 8:,}")
                    st.write(f"- Reduction: {compression_ratio:.2f}%")
                
                st.write("---")
                st.write("**Token Statistics**")
                unique_tokens = len(set(formatted_tokens))
                st.write(f"- Generated: {len(tokens)}")
                st.write(f"- Unique: {unique_tokens}")
                st.write(f"- Per Character: {len(tokens)/len(file_content):.3f}")
            
            # TAB 4: BINARY PREVIEW
            with tab4:
                st.write("### Binary Output Preview")
                st.write(f"*Showing first 64 bytes of {len(encoded)} bytes*")
                preview = encoded[:64]
                formatted_binary = ' '.join(f"{byte:08b}" for byte in preview)
                st.code(formatted_binary, language='text')
                st.info(f"Total size: {format_bytes(len(encoded))} ({len(encoded) * 8:,} bits)")
            
            # TAB 5: COMPARISON
            with tab5:
                st.write("### Before & After Comparison")
                
                comparison_data = {
                    "Metric": ["File Size", "Bits", "Compression Time"],
                    "Original": [format_bytes(file_size), f"{file_size * 8:,}", "-"],
                    "Compressed": [format_bytes(compressed_size), f"{compressed_size * 8:,}", f"{compression_time:.3f}s"]
                }
                
                st.table(comparison_data)
                
                # Visual comparison
                chart_data = pd.DataFrame({
                    "Type": ["Original", "Compressed"],
                    "Size (Bytes)": [file_size, compressed_size]
                })
                st.bar_chart(chart_data.set_index("Type"))
            
            # TAB 6: DOWNLOAD
            with tab6:
                st.write("### Download Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 Compressed Binary (.bin)",
                        data=encoded,  # Already bytes
                        file_name=f"{st.session_state['original_filename'].split('.')[0]}.bin",
                        mime="application/octet-stream"
                    )
                
                with col2:
                    # Encode codes dict as array: index = byte value, value = code
                    # This prevents key corruption during download/re-upload
                    codes_array = [''] * 256  # 256 possible byte values
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
                        mime="application/json"
                    )

# ==================== BATCH COMPRESS MODE ==================== #
elif mode == "Batch Compress":
    st.markdown("<div class='main-header'>Batch Compression</div>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload multiple files to compress",
        type=["txt", "json", "py", "csv", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("▶ Compress All Files", use_container_width=True, type="primary"):
        ensure_output_dir()
        progress_bar = st.progress(0)
        results = []
        
        for idx, file in enumerate(uploaded_files):
            with st.spinner(f"Compressing {file.name}..."):
                try:
                    file_content = file.read().decode('utf-8', errors='ignore')
                    file_size = len(file_content)
                    
                    start_time = time.time()
                    compressor = Compressor()
                    encoded, root, tokens, codes = compressor.compress(file_content)
                    compression_time = time.time() - start_time
                    
                    # encoded is now bytes (properly packed)
                    compressed_size = len(encoded)
                    compression_ratio = (1 - compressed_size / file_size) * 100 if file_size > 0 else 0
                    space_saved = file_size - compressed_size
                    
                    # Save compressed file (as binary)
                    output_path = f"{OUTPUT_DIR}/{file.name.split('.')[0]}.bin"
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    with open(output_path, 'wb') as f:
                        f.write(encoded)
                    
                    results.append({
                        "File": file.name,
                        "Original": format_bytes(file_size),
                        "Compressed": format_bytes(compressed_size),
                        "Ratio": f"{compression_ratio:.2f}%",
                        "Time": f"{compression_time:.3f}s",
                        "✅": "Done"
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
                        "File": file.name,
                        "Status": f"❌ Error: {str(e)}"
                    })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        st.divider()
        st.write("### Results")
        st.table(results)
        st.success("✅ Batch compression complete!")

# ==================== DECOMPRESS MODE ==================== #
elif mode == "Decompress":
    st.markdown("<div class='main-header'>File Decompression</div>", unsafe_allow_html=True)
    
    st.write("To decompress a file, you need:")
    st.write("1. The compressed `.bin` file")
    st.write("2. The corresponding `_metadata.json` file (contains Huffman codes)")
    
    col1, col2 = st.columns(2)
    with col1:
        bin_file = st.file_uploader("📦 Upload .bin file", type=["bin"], key="decomp_bin")
    
    with col2:
        metadata_file = st.file_uploader("📋 Upload metadata.json", type=["json"], key="decomp_meta")
    
    if bin_file and metadata_file:
        try:
            # Read the compressed binary file
            compressed_data = bin_file.read()
            
            # Read and parse the metadata
            metadata = json.load(metadata_file)
            codes_from_json = metadata.get('codes', [])
            original_filename = metadata.get('filename', 'decompressed.txt')
            
            # Validate that codes exist
            if not codes_from_json:
                st.error("❌ Metadata does not contain valid Huffman codes!")
                st.stop()
            
            print(f"[DEBUG] Codes from JSON: type={type(codes_from_json)}, length={len(codes_from_json)}")
            print(f"[DEBUG] Sample codes_from_json: {codes_from_json[:10] if isinstance(codes_from_json, list) else codes_from_json}")
            
            # Decode codes array: convert index to byte value
            # If it's already a dict ( from older format), use it directly
            if isinstance(codes_from_json, dict):
                codes_converted = {int(k): v for k, v in codes_from_json.items()}
            else:
                # It's an array - convert index to byte value
                codes_converted = {}
                try:
                    for byte_val, binary_code in enumerate(codes_from_json):
                        if binary_code:  # Only add non-empty codes
                            codes_converted[byte_val] = binary_code
                except Exception as e:
                    st.error(f"❌ Error decoding Huffman codes: {str(e)}")
                    print(f"[ERROR] Failed to decode codes: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    st.stop()
            
            print(f"[DEBUG] Codes converted: {len(codes_converted)} codes")
            print(f"[DEBUG] Sample converted codes: {list(codes_converted.items())[:5]}")
            
            if not codes_converted:
                st.error("❌ No valid Huffman codes found after conversion!")
                st.stop()
            
            st.divider()
            st.write("### Decompressing...")
            
            try:
                with st.spinner("Decompressing file..."):
                    start_time = time.time()
                    decompressor = Decompressor()
                    
                    # Call decompress with codes
                    decompressed_content = decompressor.decompress(compressed_data, codes_converted)
                    decompression_time = time.time() - start_time
                
                # Store results
                st.session_state['decompressed'] = True
                st.session_state['decompressed_content'] = decompressed_content
                st.session_state['decompressed_size'] = len(decompressed_content)
                st.session_state['decompressed_filename'] = original_filename
                st.session_state['decompression_time'] = decompression_time
                
                st.success(f"✅ Decompression Complete! ({decompression_time:.3f}s)")
            except Exception as decompression_error:
                st.error(f"❌ Decompression failed: {str(decompression_error)}")
                print(f"[ERROR] Decompression error: {str(decompression_error)}")
                import traceback
                print(traceback.format_exc())
                st.stop()
            
            st.divider()
            
            # Display decompression results
            st.subheader("Decompression Results")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Compressed Size", format_bytes(len(compressed_data)))
            with metric_col2:
                st.metric("Decompressed Size", format_bytes(len(decompressed_content)))
            with metric_col3:
                st.metric("Expansion Ratio", f"{(len(decompressed_content) / len(compressed_data)):.2f}x")
            with metric_col4:
                st.metric("Time Taken", f"{decompression_time:.3f}s")
            
            st.divider()
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Preview", "Statistics", "Download"])
            
            with tab1:
                st.write("### Decompressed File Preview")
                preview_text = decompressed_content[:500]
                if len(decompressed_content) > 500:
                    st.text_area("Content (First 500 chars):", preview_text, height=200, disabled=True)
                    st.info(f"... and {len(decompressed_content) - 500} more characters")
                else:
                    st.text_area("Full Content:", decompressed_content, height=200, disabled=True)
            
            with tab2:
                st.write("### Decompression Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Input (Compressed)**")
                    st.write(f"- Size: {format_bytes(len(compressed_data))}")
                    st.write(f"- Bytes: {len(compressed_data):,}")
                
                with col2:
                    st.write("**Output (Original)**")
                    st.write(f"- Size: {format_bytes(len(decompressed_content))}")
                    st.write(f"- Characters: {len(decompressed_content):,}")
                
                st.write("---")
                st.write("**Metadata Information**")
                st.write(f"- Original Filename: {original_filename}")
                st.write(f"- Huffman Codes: {len(codes_converted)}")
                st.write(f"- Decompression Time: {decompression_time:.4f} seconds")
            
            with tab3:
                st.write("### Download Decompressed File")
                
                # Download button
                st.download_button(
                    label="📥 Download Decompressed File",
                    data=decompressed_content,
                    file_name=original_filename,
                    mime="text/plain",
                    use_container_width=True
                )
                
                st.info(f"File will be saved as: **{original_filename}**")
        
        except json.JSONDecodeError:
            st.error("❌ Invalid metadata file! Please upload a valid JSON file.")
        except Exception as e:
            st.error(f"❌ Decompression failed: {str(e)}")
            st.write("**Troubleshooting:**")
            st.write("- Ensure you're using the correct `.bin` and `_metadata.json` files together")
            st.write("- The metadata must contain the Huffman codes from the original compression")
            st.write("- Check that the files weren't corrupted during transfer")
    elif bin_file or metadata_file:
        st.warning("⚠️ Please upload BOTH the .bin file AND the metadata.json file to decompress")


# ==================== ANALYTICS MODE ==================== #
elif mode == "Analytics":
    st.markdown("<div class='main-header'>Compression Analytics</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Algorithm", "DEFLATE")
    with col2:
        st.metric("Components", "LZ77 + Huffman")
    with col3:
        st.metric("Data Loss", "Zero")
    
    st.divider()
    
    st.write("### How DEFLATE Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Stage 1: LZ77 Dictionary**")
        st.write("- Finds repeated sequences\n- Replaces with references\n- Creates token stream")
    
    with col2:
        st.write("**Stage 2: Huffman Encoding**")
        st.write("- Analyzes token frequency\n- Builds binary tree\n- Short codes = frequent tokens")
    
    st.divider()
    
    st.write("### Expected Ratios")
    ratio_data = {
        "File Type": ["Text", "Code", "JSON", "CSV", "Compressed"],
        "Ratio": ["50-60%", "55-65%", "45-55%", "40-50%", "0-5%"]
    }
    st.table(ratio_data)

# ==================== HISTORY MODE ==================== #
elif mode == "History":
    st.markdown("<div class='main-header'>Compression History</div>", unsafe_allow_html=True)
    
    history = load_history()
    
    if history:
        # Export button
        if st.button("📊 Export as CSV"):
            df = pd.DataFrame(history)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="compression_history.csv",
                mime="text/csv"
            )
        
        st.divider()
        
        # Display statistics
        df = pd.DataFrame(history)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", len(history))
        with col2:
            st.metric("Avg Ratio", f"{df['compression_ratio'].mean():.2f}%")
        with col3:
            st.metric("Total Saved", format_bytes(df['space_saved'].sum()))
        with col4:
            st.metric("Total Time", f"{df['time_taken'].sum():.2f}s")
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Compression Ratios Over Time")
            st.line_chart(df['compression_ratio'])
        
        with col2:
            st.write("### Space Saved by File")
            chart_df = df[['filename', 'space_saved']].head(10)
            st.bar_chart(chart_df.set_index('filename')['space_saved'])
        
        st.divider()
        
        # Detailed history table
        st.write("### Recent Compressions")
        display_df = df[['filename', 'compression_ratio', 'space_saved', 'timestamp']].head(10)
        st.dataframe(display_df, use_container_width=True)
    
    else:
        st.info("📭 No compression history yet. Start compressing to see it here!")

# ==================== SETTINGS MODE ==================== #
elif mode == "Settings":
    st.markdown("<div class='main-header'>Settings</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Theme")
        if st.button(f"Current: {st.session_state.theme.upper()}"):
            st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
            st.rerun()
    
    with col2:
        st.write("### Files & Storage")
        if st.button("🗑️ Clear History"):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.success("History cleared!")
            st.rerun()
    
    st.divider()
    
    st.write("### About DEFLATE")
    st.write("""
    **DEFLATE** is a lossless data compression algorithm combining:
    - **LZ77** dictionary encoding (finds patterns)
    - **Huffman** statistical encoding (assigns codes)
    
    Used in ZIP, PNG, HTTP compression and more.
    """)
    
    st.write("### Project Info")
    st.write(f"- Mode: Unified Streamlit App\n- Version: Pro\n- Storage: {HISTORY_FILE}")
