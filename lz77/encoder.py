class LZ77Encoder:
    def __init__(self, window_size=2048, lookahead_buffer_size=258, min_match_length=4):
        """
        LZ77 Encoder with HASH-BASED matching for performance
        window_size: How far back to look for matches (reduced to 2KB for speed)
        lookahead_buffer_size: Max length of match
        min_match_length: Minimum length to encode as match
        """
        self.window_size = window_size
        self.lookahead_buffer_size = lookahead_buffer_size
        self.min_match_length = min_match_length

    def compress(self, data):
        """
        Compress data using LZ77 with hash-based matching (fast!)
        Returns: list of tokens [('L', char) or ('M', distance, length)]
        """
        i = 0
        tokens = []
        hash_table = {}  # Maps hash(3-char sequences) -> list of positions

        while i < len(data):
            # Build hash table for current window
            if i < len(data) - 2:
                # Create hash for current 3-char sequence
                seq = data[i:i+3]
                hash_key = hash(seq)
                if hash_key not in hash_table:
                    hash_table[hash_key] = []
                hash_table[hash_key].append(i)

            match_length = 0
            match_distance = 0

            # Try to find match using hash table (much faster!)
            if i < len(data) - 2:
                seq = data[i:i+3]
                hash_key = hash(seq)
                start_window = max(0, i - self.window_size)

                if hash_key in hash_table:
                    # Only check positions with matching 3-char sequence
                    for j in reversed(hash_table[hash_key]):
                        if j >= i or j < start_window:
                            continue

                        # Compare full strings
                        length = 0
                        while (
                            i + length < len(data)
                            and j + length < i  # Don't overlap
                            and data[j + length] == data[i + length]
                            and length < self.lookahead_buffer_size
                        ):
                            length += 1

                        if length >= self.min_match_length and length > match_length:
                            match_length = length
                            match_distance = i - j
                            break  # Early termination - found good match

            # Emit token
            if match_length >= self.min_match_length:
                tokens.append(('M', match_distance, match_length))
                i += match_length
            else:
                tokens.append(('L', data[i]))
                i += 1

        return tokens