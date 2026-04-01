from lz77.encoder import LZ77Encoder
from huffman.encoder import HuffmanEncoder
from utils.simple_token_format import SimpleTokenFormat


class Compressor:
    def __init__(self):
        self.lz77 = LZ77Encoder()
        self.huffman = HuffmanEncoder()
        self.token_encoder = SimpleTokenFormat()

    def compress(self, data):
        """
        Complete compression pipeline:
        1. LZ77 tokenization
        2. Convert tokens to binary representation (safe for all byte values)
        3. Huffman statistical encoding
        """
        # Step 1: LZ77 tokenization
        tokens = self.lz77.compress(data)

        # Step 2: Convert tokens to binary format using variable-length encoding
        # This safely handles all byte values including commas, colons, etc.
        token_bytes = self.token_encoder.encode_tokens(tokens)

        # Step 3: Huffman Encoding on token bytes
        # Convert bytes to string for Huffman encoding
        token_string = ''.join(chr(b) for b in token_bytes)
        encoded_data, root, codes = self.huffman.encode(token_string)

        # Return: encoded_binary, huffman_tree, original_tokens (for visualization), huffman_codes
        return encoded_data, root, tokens, codes