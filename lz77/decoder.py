class LZ77Decoder:
    def decompress(self, tokens):
        print(f"[DEBUG] LZ77 Decoder starting with {len(tokens)} tokens")
        output = []
        
        token_count = 0
        for token in tokens:
            token_count += 1
            if token_count % 1000 == 0:
                print(f"[DEBUG] Processed {token_count} tokens, output size: {len(output)}")
            
            if token[0] == 'L':
                # Literal
                output.append(token[1])
            elif token[0] == 'M':
                # Match (distance, length)
                _, distance, length = token
                
                if distance <= 0:
                    print(f"[ERROR] Invalid distance {distance}")
                    raise ValueError(f"Invalid distance: {distance} must be > 0")

                if distance > len(output):
                    print(f"[ERROR] Invalid distance {distance}, output size: {len(output)}")
                    raise ValueError(f"Invalid distance: {distance} > {len(output)}")

                start = len(output) - distance

                for i in range(length):
                    # In LZ77, length can be greater than distance (repeating pattern)
                    # We use modulo or simply access the growing output list
                    output.append(output[start + (i % distance)])
            else:
                print(f"[WARNING] Unknown token type: {token[0]}")
        
        result = ''.join(output)
        print(f"[DEBUG] LZ77 Decoder complete. Output size: {len(result)}")
        return result