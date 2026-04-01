from lz77.decoder import LZ77Decoder
from huffman.decoder import HuffmanDecoder
from utils.simple_token_format import SimpleTokenFormat
import json


class Decompressor:
    def __init__(self):
        self.lz77 = LZ77Decoder()
        self.huffman = HuffmanDecoder()
        self.token_encoder = SimpleTokenFormat()

    def decompress(self, encoded_data, codes):
        """
        Decompress using Huffman codes
        
        Args:
            encoded_data: bytes (packed binary from compressor)
            codes: dict mapping integer byte values (0-255) to binary codes
        
        Returns:
            original_data: decompressed string
        """
        if not encoded_data or not codes:
            return ""
        
        try:
            print(f"[DEBUG] Starting decompression. Data size: {len(encoded_data)}, Codes: {len(codes)}")
            print(f"[DEBUG] Codes type: {type(codes)}, Sample codes: {list(codes.items())[:5] if codes else 'empty'}")
            
            # Step 1: Huffman Decode (codes dict has integer keys)
            print(f"[DEBUG] Step 1: Huffman decoding...")
            decoded_bytes = self.huffman.decode_to_bytes(encoded_data, codes)
            print(f"[DEBUG] Huffman decoded bytes length: {len(decoded_bytes)}")
            print(f"[DEBUG] First 20 decoded bytes: {decoded_bytes[:20] if decoded_bytes else 'empty'}")
            
            if not decoded_bytes:
                raise ValueError("Huffman decoding produced empty result")
            
            # Step 2: Decode token bytes back to tokens using TokenEncoder
            print(f"[DEBUG] Step 2: Decoding tokens...")
            print(f"[DEBUG] Input to decode_tokens: {len(decoded_bytes)} bytes")
            tokens = self.token_encoder.decode_tokens(decoded_bytes)
            print(f"[DEBUG] Tokens decoded: {len(tokens)} tokens")
            print(f"[DEBUG] Sample tokens: {tokens[:5] if tokens else 'no tokens'}")
            
            if not tokens:
                raise ValueError("Token decoding produced no tokens")
            
            # Step 3: LZ77 Decode
            print(f"[DEBUG] Step 3: LZ77 decompression...")
            original_data = self.lz77.decompress(tokens)
            print(f"[DEBUG] Decompression complete. Output size: {len(original_data)}")
            
            return original_data
        except IndexError as e:
            raise ValueError(f"Index error during decompression (corrupted file?): {str(e)}")
        except Exception as e:
            import traceback
            print(f"[ERROR] {traceback.format_exc()}")
            raise ValueError(f"Decompression failed: {str(e)}")