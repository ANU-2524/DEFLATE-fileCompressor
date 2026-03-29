class HuffmanDecoder:
    def decode(self, encoded_data, codes):
        # Step 1: Reverse the codes
        reverse_codes = {v: k for k, v in codes.items()}

        decoded_output = []
        current_code = ""

        # Step 2: Read bit by bit
        for bit in encoded_data:
            current_code += bit

            # If code matches
            if current_code in reverse_codes:
                decoded_output.append(reverse_codes[current_code])
                current_code = ""

        return ''.join(decoded_output)