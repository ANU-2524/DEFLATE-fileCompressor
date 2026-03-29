from lz77.decoder import LZ77Decoder
from huffman.decoder import HuffmanDecoder


class Decompressor:
    def __init__(self):
        self.lz77 = LZ77Decoder()
        self.huffman = HuffmanDecoder()

    # Convert string → tokens
    def string_to_tokens(self, data):
        tokens = []
        parts = data.split()

        for p in parts:
            if p.startswith('L|'):
                tokens.append(('L', p[2:]))
            else:
                d, l = p[2:].split(',')
                tokens.append(('M', int(d), int(l)))

        return tokens

    def decompress(self, encoded_data, codes, tokens=None):
        """
        encoded_data → binary string
        codes → Huffman code map (char → binary)
        tokens → optional (not needed ideally, but kept for safety)
        """

        # Step 1: Huffman Decode (using codes)
        decoded_string = self.huffman.decode(encoded_data, codes)

        # Step 2: String → LZ77 Tokens
        tokens = self.string_to_tokens(decoded_string)

        # Step 3: LZ77 Decode
        original_data = self.lz77.decompress(tokens)

        return original_data