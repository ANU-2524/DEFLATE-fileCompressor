import heapq
from collections import Counter


class Node:
    def __init__(self, byte_val, freq):
        self.byte_val = byte_val  # Integer 0-255 or None for internal nodes
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanEncoder:
    def build_tree(self, data_bytes):
        """Build Huffman tree from list of byte values (integers 0-255)"""
        freq_map = Counter(data_bytes)
        
        heap = [Node(byte_val, freq) for byte_val, freq in freq_map.items()]
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
        """Build Huffman codes - with integer byte values as keys"""
        codes = {}  # Maps integer byte value to binary code string

        def generate(node, current_code):
            if node is None:
                return

            if node.byte_val is not None:
                # Leaf node - store the code with integer key
                codes[node.byte_val] = current_code if current_code else "0"
                return

            generate(node.left, current_code + "0")
            generate(node.right, current_code + "1")

        generate(root, "")
        return codes

    def encode(self, text):
        """
        Huffman encoding with proper binary packing
        text: input string
        Returns: (binary_data, root, codes where keys are integers 0-255)
        """
        # Step 1: Build tree from byte values
        if not text:
            return b"", None, {}
        
        # Convert string to byte values
        data_bytes = [ord(c) for c in text]
        
        root = self.build_tree(data_bytes)
        if not root:
            return b"", None, {}

        # Step 2: Build codes (with integer keys)
        codes = self.build_codes(root)
        
        print(f"[DEBUG] Huffman encoder - codes created with integer keys: {len(codes)} unique")
        print(f"[DEBUG] Sample codes: {list(codes.items())[:3]}")

        # Step 3: Encode text to binary string using integer keys
        binary_string = ''.join(codes[byte_val] for byte_val in data_bytes)

        # Step 4: Pack binary string into actual bytes
        packed_data = self._pack_binary(binary_string)

        # Return packed bytes, root, and codes (with integer keys)
        return packed_data, root, codes
    
    def _pack_binary(self, binary_str):
        """
        Convert binary string to actual bytes (8 bits per byte)
        Returns bytes with padding info in first byte
        """
        if not binary_str:
            return b'\x00'
        
        # Calculate padding needed to make length multiple of 8
        padding = (8 - len(binary_str) % 8) % 8
        binary_str_padded = binary_str + "0" * padding
        
        # Convert to bytes
        byte_list = []
        for i in range(0, len(binary_str_padded), 8):
            byte = binary_str_padded[i:i+8]
            byte_list.append(int(byte, 2))
        
        # Store padding info in first byte
        # Format: [padding_bits (8 bits)] [actual compressed data...]
        data = bytes([padding] + byte_list)
        
        return data