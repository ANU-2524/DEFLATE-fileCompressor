from lz77.encoder import LZ77Encoder
from lz77.decoder import LZ77Decoder

data = "helloo , helloo , Hi !!"
encoder = LZ77Encoder()
tokens = encoder.compress(data) 

print('Tokens : ' , tokens)

decoder = LZ77Decoder()
decoded = decoder.decompress(tokens)

print("Decoded : " , decoded)