import os
from typing import BinaryIO
import regex as re
from collections import Counter
from itertools import repeat


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi:] = [file_size] * (len(chunk_boundaries) - bi)
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
            
        if chunk_boundaries[bi] == file_size:
            break
        
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(file_path, start, end, special_tokens):
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        
        counts = Counter()
        if special_tokens:
            special_pattern = "|".join(re.escape(s) for s in special_tokens)
            mini_chunks = re.split(special_pattern, chunk)
        else:
            mini_chunks = [chunk]
        for mini_chunk in mini_chunks:
            for token in iter(re.finditer(PAT, mini_chunk)):
                counts[token.group(0)] += 1
        
        return counts

def pretokenize(file_path, special_tokens):
    num_processes = 16
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    from concurrent.futures import ProcessPoolExecutor
        
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        ret_list = list(executor.map(process_chunk, repeat(file_path), boundaries[:-1], boundaries[1:], repeat(special_tokens)))
    
    ret = Counter()
    for s in ret_list:
        ret.update(s)
    
    return ret

## Usage
if __name__ == "__main__":
    file_path = "data/TinyStoriesV2-GPT4-train.txt"
    ret = pretokenize(file_path, ["<|endoftext|>"])
    print(type(ret))
    print(ret.most_common(20))