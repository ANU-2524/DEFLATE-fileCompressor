class LZ77Decoder:
    def decompress(self, tokens):
        output = []

        for token in tokens:
            if token[0] == 'L':
                # Literal
                output.append(token[1])
            else:
                # Match (distance, length)
                _, distance, length = token

                start = len(output) - distance

                for i in range(length):
                    output.append(output[start + i])

        return ''.join(output)