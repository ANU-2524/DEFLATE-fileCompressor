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
            else:
                # Match (distance, length)
                _, distance, length = token
                
                if distance > len(output):
                    print(f"[ERROR] Invalid distance {distance}, output size: {len(output)}")
                    raise ValueError(f"Invalid distance: {distance} > {len(output)}")

                start = len(output) - distance

                for i in range(length):
                    if start + i >= len(output):
                        print(f"[ERROR] Invalid index: start={start}, i={i}, output_size={len(output)}")
                        raise IndexError(f"Invalid index in match token")
                    output.append(output[start + i])
        
        result = ''.join(output)
        print(f"[DEBUG] LZ77 Decoder complete. Output size: {len(result)}")
        return result