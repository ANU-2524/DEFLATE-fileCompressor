import networkx as nx
import matplotlib.pyplot as plt

class Visualizer:

    @staticmethod
    def draw_huffman_tree(root):
        G = nx.DiGraph()

        def add_edges(node, parent=None):
            if node:
                # Add node label
                label = node.char if node.char else str(node.freq)
                G.add_node(id(node), label=label)

                if parent:
                    G.add_edge(id(parent), id(node))

                add_edges(node.left, node)
                add_edges(node.right, node)

        add_edges(root)

        pos = nx.spring_layout(G)

        labels = nx.get_node_attributes(G, 'label')

        nx.draw(G, pos, labels=labels, with_labels=True)
        plt.title("Huffman Tree")
        plt.show()