#!/usr/bin/env python
"""Quick test of compression/decompression pipeline"""

from utils.simple_token_format import SimpleTokenFormat
from huffman.encoder import HuffmanEncoder
from huffman.decoder import HuffmanDecoder

def test_token_format():
    print("=" * 50)
    print("TEST 1: Token Format Encoding/Decoding")
    print("=" * 50)
    
    encoder = SimpleTokenFormat()
    tokens = [('L', 'H'), ('L', 'e'), ('M', 10, 5), ('L', 'l')]
    
    encoded = encoder.encode_tokens(tokens)
    decoded = encoder.decode_tokens(encoded)
    
    if tokens == decoded:
        print("✅ Token format: WORKS")
        return True
    else:
        print("❌ Token format: FAILED")
        return False

def test_huffman_roundtrip():
    print("\n" + "=" * 50)
    print("TEST 2: Huffman Roundtrip (Bytes Preservation)")
    print("=" * 50)
    
    encoder = HuffmanEncoder()
    decoder = HuffmanDecoder()
    token_fmt = SimpleTokenFormat()
    
    # Create diverse test tokens
    tokens = [('L', 'a')] * 100 + [('L', 'b')] * 50 + [('M', 100, 10)] * 25
    
    # Encode tokens
    token_bytes = token_fmt.encode_tokens(tokens)
    print(f"Token bytes: {len(token_bytes)} bytes")
    
    # Huffman encode
    token_string = ''.join(chr(b) for b in token_bytes)
    huff_data, root, codes = encoder.encode(token_string)
    print(f"Huffman encoded: {len(huff_data)} bytes (ratio: {len(huff_data)/len(token_bytes):.2%})")
    
    # Huffman decode to bytes
    decoded_bytes = decoder.decode_to_bytes(huff_data, codes)
    print(f"Decoded bytes: {len(decoded_bytes)} bytes")
    
    # Verify
    if token_bytes == decoded_bytes:
        print("✅ Huffman roundtrip: WORKS")
        return True
    else:
        print(f"❌ Huffman roundtrip: FAILED")
        print(f"   Original: {len(token_bytes)} bytes")
        print(f"   Decoded:  {len(decoded_bytes)} bytes")
        return False

def test_full_pipeline():
    print("\n" + "=" * 50)
    print("TEST 3: Full Decompression Pipeline")
    print("=" * 50)
    
    encoder = HuffmanEncoder()
    decoder = HuffmanDecoder()
    token_fmt = SimpleTokenFormat()
    
    # Create test tokens
    tokens = [('L', 'H'), ('L', 'e'), ('L', 'l'), ('L', 'l'), ('L', 'o')]
    
    # Encode
    token_bytes = token_fmt.encode_tokens(tokens)
    token_string = ''.join(chr(b) for b in token_bytes)
    huff_data, root, codes = encoder.encode(token_string)
    
    # Decode
    decoded_bytes = decoder.decode_to_bytes(huff_data, codes)
    decoded_tokens = token_fmt.decode_tokens(decoded_bytes)
    
    if tokens == decoded_tokens:
        print("✅ Full pipeline: WORKS")
        return True
    else:
        print("❌ Full pipeline: FAILED")
        print(f"Input tokens:  {tokens}")
        print(f"Output tokens: {decoded_tokens}")
        return False

if __name__ == "__main__":
    results = []
    results.append(test_token_format())
    results.append(test_huffman_roundtrip())
    results.append(test_full_pipeline())
    
    print("\n" + "=" * 50)
    print(f"SUMMARY: {sum(results)}/{len(results)} tests passed")
    print("=" * 50)
    
    if all(results):
        print("✅ ALL TESTS PASSED - Project is READY to use!")
    else:
        print("❌ SOME TESTS FAILED - Do NOT run app yet")
