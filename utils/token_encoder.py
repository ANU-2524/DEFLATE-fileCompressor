"""
Binary token encoder - encodes LZ77 tokens efficiently as binary
"""

class TokenEncoder:
    """
    Encode tokens as binary using variable-length integers
    Much more efficient than string representation
    """
    
    def encode_tokens(self, tokens):
        """
        Encode tokens to binary bytes
        Format:
        - Literal: [0][8-bit char code]
        - Match: [1][variable-int distance][variable-int length]
        """
        bits = ""
        
        for token in tokens:
            if token[0] == 'L':
                # Literal: bit 0 + 8 bits for character
                char_code = ord(token[1])
                bits += "0" + format(char_code, '08b')
            else:
                # Match: bit 1 + variable-int distance + variable-int length
                distance = token[1]
                length = token[2]
                bits += "1"
                bits += self._encode_varint(distance)
                bits += self._encode_varint(length)
        
        # Pad to byte boundary
        padding = (8 - len(bits) % 8) % 8
        bits += "0" * padding
        
        # Convert to bytes with padding info
        byte_list = []
        for i in range(0, len(bits), 8):
            byte_val = int(bits[i:i+8], 2)
            byte_list.append(byte_val)
        
        # First byte = padding info
        result = bytes([padding] + byte_list)
        return result
    
    def _encode_varint(self, num):
        """
        Encode number as variable-length integer (variable-length quantity)
        Efficient for small numbers
        """
        bits = ""
        while num >= 128:
            bits += format((num % 128) | 128, '08b')
            num //= 128
        bits += format(num, '08b')
        return bits
    
    def decode_tokens(self, data):
        """
        Decode binary bytes back to tokens
        """
        if not data or len(data) < 2:
            return []
        
        # Extract padding info
        padding = data[0]
        
        # Convert bytes to binary string
        bits = ""
        for byte in data[1:]:
            bits += format(byte, '08b')
        
        # Remove padding
        if padding > 0:
            bits = bits[:-padding]
        
        # Decode tokens
        tokens = []
        pos = 0
        
        while pos < len(bits):
            if pos + 1 > len(bits):
                break
            
            token_type = bits[pos]
            pos += 1
            
            if token_type == '0':
                # Literal
                if pos + 8 > len(bits):
                    break
                char_code = int(bits[pos:pos+8], 2)
                tokens.append(('L', chr(char_code)))
                pos += 8
            else:
                # Match
                distance, pos = self._decode_varint(bits, pos)
                length, pos = self._decode_varint(bits, pos)
                tokens.append(('M', distance, length))
        
        return tokens
    
    def _decode_varint(self, bits, pos):
        """
        Decode variable-length integer
        """
        num = 0
        multiplier = 1
        
        while pos < len(bits):
            if pos + 8 > len(bits):
                break
            
            byte_val = int(bits[pos:pos+8], 2)
            pos += 8
            
            num += (byte_val & 0x7F) * multiplier
            
            if byte_val < 128:
                break
            
            multiplier *= 128
        
        return num, pos
