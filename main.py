# from lz77.encoder import LZ77Encoder
# from lz77.decoder import LZ77Decoder

# data = "helloo , helloo , Hi !!"
# encoder = LZ77Encoder()
# tokens = encoder.compress(data) 

# print('Tokens : ' , tokens)

# decoder = LZ77Decoder()
# decoded = decoder.decompress(tokens)

# print("Decoded : " , decoded)

from huffman.encoder import HuffmanEncoder
from huffman.decoder import HuffmanDecoder

data = "hello huffman"

encoder = HuffmanEncoder()
encoded_data, root = encoder.encode(data)

print("Encoded:", encoded_data)

decoder = HuffmanDecoder()
decoded_data = decoder.decode(encoded_data, root)

print("Decoded:", decoded_data)