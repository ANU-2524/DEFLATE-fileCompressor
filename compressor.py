from lz77.encoder import LZ77Encoder
from huffman.encoder import HuffmanEncoder


class Compressor:
    def __init__(self):
        self.lz77 = LZ77Encoder()
        self.huffman = HuffmanEncoder()

    # Convert tokens → string
    def tokens_to_string(self, tokens):
        result = []
        for t in tokens:
            if t[0] == 'L':
                result.append(f"L|{t[1]}")
            else:
                result.append(f"M|{t[1]},{t[2]}")
        return ' '.join(result)

    def compress(self, data):
        # Step 1: LZ77
        tokens = self.lz77.compress(data)

        # Step 2: Tokens → String
        token_string = self.tokens_to_string(tokens)

        # Step 3: Huffman Encoding
        encoded_data, root, codes = self.huffman.encode(token_string)

        # IMPORTANT: return codes also (for decoding)
        return encoded_data, root, tokens, codes