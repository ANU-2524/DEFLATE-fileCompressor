class LZ77Encoder:
    def __init__(self, window_size=20, lookahead_buffer_size=15):
        self.window_size = window_size
        self.lookahead_buffer_size = lookahead_buffer_size

    def compress(self, data):
        i = 0
        tokens = []

        while i < len(data):
            match_length = 0
            match_distance = 0

            # Sliding window start
            start_window = max(0, i - self.window_size)

            # Try to find longest match
            for j in range(start_window, i):
                length = 0

                while (
                    i + length < len(data)
                    and data[j + length] == data[i + length]
                    and length < self.lookahead_buffer_size
                ):
                    length += 1

                if length > match_length:
                    match_length = length
                    match_distance = i - j

            # If match found
            if match_length > 0:
                tokens.append(('M', match_distance, match_length))
                i += match_length
            else:
                tokens.append(('L', data[i]))
                i += 1

        return tokens