"""
Bulletproof Token Encoder - Fixed version with better format
Uses simpler, more reliable binary encoding
"""

class TokenEncoderV2:
    """
    Encode tokens using a simpler, more reliable binary format
    Uses fixed-size fields instead of variable-length integers
    """
    
    def encode_tokens(self, tokens):
        """
        Encode tokens to binary bytes
        Format:
        - Literal: [0][8-bit char code]
        - Match: [1][16-bit distance][16-bit length]
        
        Much simpler than varint - no corruption issues
        """
        bits = ""
        
        for token in tokens:
            if token[0] == 'L':
                # Literal: bit 0 + 8 bits for character
                char_code = ord(token[1])
                bits += "0" + format(char_code, '08b')
            else:
                # Match: bit 1 + 16-bit distance + 16-bit length
                distance = min(token[1], 65535)  # Cap at 16-bit max
                length = min(token[2], 65535)      # Cap at 16-bit max
                bits += "1"
                bits += format(distance, '016b')
                bits += format(length, '016b')
        
        # Pad to byte boundary
        padding = (8 - len(bits) % 8) % 8
        bits += "0" * padding
        
        # Convert to bytes
        byte_list = []
        for i in range(0, len(bits), 8):
            byte_val = int(bits[i:i+8], 2)
            byte_list.append(byte_val)
        
        # Format: [padding (8-bit)][data...]
        # Keep it simple - no complex headers
        result = bytes([padding]) + bytes(byte_list)
        
        return result
    
    def decode_tokens(self, data):
        """
        Decode binary bytes back to tokens
        Simple format: padding byte followed by token data
        """
        if not data or len(data) < 1:
            raise ValueError(f"Invalid token data: too short ({len(data)} bytes)")
        
        try:
            # Extract padding
            padding = data[0]
            
            if padding >= 8:
                raise ValueError(f"Invalid padding: {padding}")
            
            # Extract actual data (everything after padding byte)
            token_data = data[1:]
            
            # Convert to binary string
            bits = ""
            for byte in token_data:
                bits += format(byte, '08b')
            
            # Remove padding
            if padding > 0:
                bits = bits[:-padding]
            
            # Decode tokens
            tokens = []
            pos = 0
            token_count = 0
            
            while pos < len(bits):
                if pos + 1 > len(bits):
                    break
                
                token_type = bits[pos]
                pos += 1
                
                if token_type == '0':
                    # Literal: 0 + 8 bits
                    if pos + 8 > len(bits):
                        raise ValueError(f"Incomplete literal token at pos {pos}")
                    
                    char_code = int(bits[pos:pos+8], 2)
                    tokens.append(('L', chr(char_code)))
                    pos += 8
                    token_count += 1
                else:
                    # Match: 1 + 16-bit distance + 16-bit length
                    if pos + 32 > len(bits):
                        raise ValueError(f"Incomplete match token at pos {pos}")
                    
                    distance = int(bits[pos:pos+16], 2)
                    length = int(bits[pos+16:pos+32], 2)
                    
                    if distance <= 0 or length <= 0:
                        raise ValueError(f"Invalid match token: distance={distance}, length={length}")
                    
                    tokens.append(('M', distance, length))
                    pos += 32
                    token_count += 1
            
            return tokens
        
        except Exception as e:
            raise ValueError(f"Token decoding failed: {str(e)}")
