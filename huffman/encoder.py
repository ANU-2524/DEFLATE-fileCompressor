import heapq
from collections import Counter


class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanEncoder:
    def build_tree(self, text):
        freq_map = Counter(text)

        heap = [Node(char, freq) for char, freq in freq_map.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)

            merged = Node(None, left.freq + right.freq)
            merged.left = left
            merged.right = right

            heapq.heappush(heap, merged)
        if not heap:
            return None
        return heap[0]

    def build_codes(self, root):
        codes = {}

        def generate(node, current_code):
            if node is None:
                return

            if node.char is not None:
                codes[node.char] = current_code
                return

            generate(node.left, current_code + "0")
            generate(node.right, current_code + "1")

        generate(root, "")
        return codes

    def encode(self, text):
        # Step 1: Build tree
        if not text:
            return "", None, {}
        root = self.build_tree(text)

        # Step 2: Build codes
        codes = self.build_codes(root)

        # Step 3: Encode text
        encoded_data = ''.join(codes[char] for char in text)

        # 🔥 IMPORTANT: return codes also
        return encoded_data, root, codes