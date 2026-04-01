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

        # Use graphviz layout for better hierarchical tree visualization
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            # Fallback to custom hierarchical layout if graphviz not available
            pos = Visualizer._hierarchy_pos(G, root)

        labels = nx.get_node_attributes(G, 'label')

        nx.draw(G, pos, labels=labels, with_labels=True, 
                node_color='lightblue', node_size=1500, 
                font_size=10, font_weight='bold', 
                arrows=True, arrowsize=20)
        plt.title("Huffman Tree (Hierarchical Layout)")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        """Fallback hierarchical layout when graphviz is not available"""
        pos = {root: (xcenter, vert_loc)}
        neighbors = list(G.neighbors(root))
        
        if len(neighbors) != 0:
            dx = width / len(neighbors)
            nextx = xcenter - width/2 - dx/2
            for neighbor in neighbors:
                nextx += dx
                pos.update(Visualizer._hierarchy_pos(G, neighbor, width=dx, 
                                                    vert_gap=vert_gap,
                                                    vert_loc=vert_loc-vert_gap, 
                                                    xcenter=nextx))
        return pos