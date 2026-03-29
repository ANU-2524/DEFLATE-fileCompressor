class HuffmanDecoder:
    def decode(self, encoded_data, root):
        decoded_output = []
        current = root

        for bit in encoded_data:
            # Traverse tree
            if bit == '0':
                current = current.left
            else:
                current = current.right

            # If leaf node → append character
            if current.left is None and current.right is None:
                decoded_output.append(current.char)
                current = root  # reset for next character

        return ''.join(decoded_output)