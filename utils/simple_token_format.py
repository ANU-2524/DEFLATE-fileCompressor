"""
Ultra-simple, bulletproof token format.
Direct binary encoding - no complexity, no headers, just pure binary.
"""

class SimpleTokenFormat:
    """
    Encode/decode tokens to/from raw bytes
    Format is extremely simple and robust
    """
    
    def encode_tokens(self, tokens):
        """
        Encode tokens to bytes
        Format per token:
        - Literal (L): [0x00][byte_value] = 2 bytes
        - Match (M): [0x01][distance_high][distance_low][length_high][length_low] = 5 bytes
        
        Simple fixed format - no variable length nonsense
        """
        bits = ""
        
        for token in tokens:
            if token[0] == 'L':
                # Literal: 0x00 marker + 8 bits for character
                char_code = ord(token[1])
                bits += "00000000"  # 0x00 marker
                bits += format(char_code, '08b')
            else:
                # Match: 0x01 marker + 16-bit distance + 16-bit length
                distance = min(token[1], 65535)
                length = min(token[2], 65535)
                bits += "00000001"  # 0x01 marker
                bits += format(distance, '016b')
                bits += format(length, '016b')
        
        # Pad to byte boundary
        padding = (8 - len(bits) % 8) % 8
        bits += "0" * padding
        
        # Convert to bytes
        result_bytes = bytearray()
        for i in range(0, len(bits), 8):
            byte_val = int(bits[i:i+8], 2)
            result_bytes.append(byte_val)
        
        return bytes(result_bytes)
    
    def decode_tokens(self, data):
        """
        Decode bytes back to tokens
        """
        if not data:
            return []
        
        # Convert bytes to bits
        bits = ""
        for byte in data:
            bits += format(byte, '08b')
        
        print(f"[DEBUG] decode_tokens: input data length={len(data)}")
        print(f"[DEBUG] First 10 bytes: {list(data[:10])}")
        print(f"[DEBUG] Binary bits length: {len(bits)}")
        print(f"[DEBUG] First 32 bits (first 4 bytes): {bits[:32] if len(bits) >= 32 else bits}")
        
        tokens = []
        pos = 0
        
        while pos + 8 <= len(bits):
            marker = bits[pos:pos+8]
            
            if marker == '00000000':
                # Literal: read 8 bits for character
                if pos + 16 > len(bits):
                    break
                pos += 8
                char_code = int(bits[pos:pos+8], 2)
                tokens.append(('L', chr(char_code)))
                pos += 8
            
            elif marker == '00000001':
                # Match: read 16-bit distance + 16-bit length
                if pos + 40 > len(bits):  # 8 (marker) + 16 (distance) + 16 (length)
                    break
                pos += 8
                distance = int(bits[pos:pos+16], 2)
                length = int(bits[pos+16:pos+32], 2)
                
                if distance <= 0 or length <= 0:
                    raise ValueError(f"Invalid match: distance={distance}, length={length}")
                
                tokens.append(('M', distance, length))
                pos += 32
            
            else:
                # Unknown marker - might be padding, stop
                print(f"[DEBUG] Unknown marker at pos {pos}: {marker}")
                break
        
        print(f"[DEBUG] Decoded {len(tokens)} tokens from {len(data)} bytes")
        if not tokens:
            print(f"[WARNING] No tokens decoded!")
            print(f"[WARNING] Data bytes: {list(data)}")
            print(f"[WARNING] After converting to binary, first marker: {bits[:8] if len(bits) >= 8 else 'too short'}")
        
        return tokens
