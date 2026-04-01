class HuffmanDecoder:
    def decode(self, encoded_data, codes):
        """
        Huffman decoding
        encoded_data: bytes (packed binary from encoder)
        codes: dict mapping integer byte values to binary codes
        Returns: decoded string (from byte values)
        """
        if not encoded_data or not codes:
            return ""
        
        print(f"[DEBUG] Huffman decoder starting. Codes count: {len(codes)}")
        
        # Validate codes - keys should be integers (byte values)
        for byte_val, code in list(codes.items())[:5]:
            print(f"[DEBUG] Sample code: byte_value={byte_val} (type={type(byte_val).__name__}) -> {code}")
        
        # Convert bytes back to binary string
        binary_string = self._unpack_binary(encoded_data)
        print(f"[DEBUG] Binary string length: {len(binary_string)}")
        
        # Reverse the codes - creates mapping from binary code to byte value
        reverse_codes = {v: k for k, v in codes.items()}
        
        print(f"[DEBUG] Reverse codes count: {len(reverse_codes)}")
        print(f"[DEBUG] Sample reverse codes: {list(reverse_codes.items())[:3]}")

        decoded_bytes = []
        current_code = ""

        # Read bit by bit
        for bit_idx, bit in enumerate(binary_string):
            current_code += bit

            # If code matches, add byte value to output
            if current_code in reverse_codes:
                byte_val = reverse_codes[current_code]
                decoded_bytes.append(byte_val)
                current_code = ""
            # Safety check: if code is getting too long, something is wrong
            elif len(current_code) > 32:
                # Too long - must be an error or corrupted data
                print(f"[ERROR] Code too long: {current_code}")
                print(f"[ERROR] Available codes: {list(reverse_codes.keys())[:10]}")
                raise ValueError(f"Huffman decoding error: code too long '{current_code[:50]}'")

        # Any remaining bits should be padding (all zeros)
        if current_code:
            # Check if remaining bits are all zeros (valid padding)
            if current_code != "0" * len(current_code):
                print(f"[WARNING] Non-zero padding bits: {current_code} (treating as padding)")

        # Convert byte values back to string
        result = ''.join(chr(b) for b in decoded_bytes)
        print(f"[DEBUG] Huffman decode complete. Output length: {len(result)}")
        return result
    
    def decode_to_bytes(self, encoded_data, codes):
        """
        Huffman decoding directly to bytes (avoiding chr/ord conversion)
        encoded_data: bytes (packed binary from encoder)
        codes: dict mapping integer byte values (0-255) to binary codes
        Returns: bytes (direct byte values, no string conversion)
        """
        if not encoded_data or not codes:
            return b""
        
        # Convert bytes back to binary string
        binary_string = self._unpack_binary(encoded_data)
        
        # Reverse the codes - creates mapping from binary code to byte value
        reverse_codes = {v: k for k, v in codes.items()}

        decoded_bytes = []
        current_code = ""

        # Read bit by bit
        for bit_idx, bit in enumerate(binary_string):
            current_code += bit

            # If code matches, add byte value to output
            if current_code in reverse_codes:
                byte_val = reverse_codes[current_code]
                decoded_bytes.append(byte_val)
                current_code = ""
            # Safety check: if code is getting too long, something is wrong
            elif len(current_code) > 32:
                # Too long - must be an error or corrupted data
                raise ValueError(f"Huffman decoding error: code too long")

        # Any remaining bits should be padding (all zeros)
        if current_code:
            # Remaining bits are padding, ignore
            pass

        # Convert byte values directly to bytes
        result = bytes(decoded_bytes)
        return result
    
    def _unpack_binary(self, data):
        """
        Convert packed bytes back to binary string
        First byte contains padding information
        """
        if not data:
            return ""
        
        # First byte contains padding info
        padding = data[0]
        
        # Convert remaining bytes to binary string
        binary_string = ""
        for byte in data[1:]:
            binary_string += format(byte, '08b')
        
        # Remove padding bits
        if padding > 0:
            binary_string = binary_string[:-padding]
        
        return binary_string