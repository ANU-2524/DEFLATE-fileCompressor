import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from compressor import Compressor


# ------------------ Huffman Tree Visualization ------------------ #
def draw_huffman_tree(root):
    G = nx.DiGraph()

    def add_edges(node, parent=None):
        if node:
            label = node.char if node.char else str(node.freq)
            G.add_node(id(node), label=label)

            if parent:
                edge_label = '0' if parent.left == node else '1'
                G.add_edge(id(parent), id(node), label=edge_label)

            add_edges(node.left, node)
            add_edges(node.right, node)

    add_edges(root)

    pos = nx.kamada_kawai_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')

    fig, ax = plt.subplots()
    nx.draw(G, pos, labels=labels, with_labels=True, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    return fig


# ------------------ Streamlit UI ------------------ #
st.set_page_config(page_title="DEFLATE Visualizer", layout="wide")

st.title("🚀 DEFLATE Compression Visualizer")
st.write("LZ77 + Huffman Coding (Real Compression Pipeline)")

file = st.file_uploader("📂 Upload a text file", type=["txt"])

if file:
    data = file.read().decode()

    compressor = Compressor()
    encoded, root, tokens , codes = compressor.compress(data)

    # ------------------ Layout ------------------ #
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Original Data")
        st.text_area("Input Text", data, height=200)

    with col2:
        st.subheader("📦 Encoded Bitstream (Preview)")
        st.text_area("Binary Output", encoded[:500] + "...", height=200)

    # ------------------ LZ77 Tokens ------------------ #
    st.subheader("🔵 LZ77 Tokens")

    formatted_tokens = []
    for t in tokens:
        if t[0] == 'L':
            formatted_tokens.append(f"L:{t[1]}")
        else:
            formatted_tokens.append(f"M:{t[1]},{t[2]}")

    st.write(formatted_tokens)

    # ------------------ Compression Stats ------------------ #
    st.subheader("📊 Compression Stats")

    original_size = len(data) * 8  # bits
    compressed_size = len(encoded)

    ratio = (1 - compressed_size / original_size) * 100 if original_size else 0

    st.write(f"Original Size: {original_size} bits")
    st.write(f"Compressed Size: {compressed_size} bits")
    st.write(f"Compression Ratio: {ratio:.2f}%")

    # ------------------ Huffman Tree ------------------ #
    st.subheader("🌳 Huffman Tree Visualization")

    fig = draw_huffman_tree(root)
    st.pyplot(fig)

    # ------------------ Extra Insight ------------------ #
    st.subheader("🧠 Insights")

    st.write(f"Total Tokens Generated: {len(tokens)}")
    st.write(f"Unique Symbols After LZ77: {len(set(formatted_tokens))}")