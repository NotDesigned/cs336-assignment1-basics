from pretokenization import pretokenize
from collections import Counter

class BPE_Tokenizer:
    def __init__(self, file_path, vocab_size, special_tokens):
        self.file_path = file_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.count = None
    
    def pretokenize(self):
        self.count = pretokenize(self.file_path, special_tokens=self.special_tokens)
     
    def train(self):
        """
        Train the Tokenizer and return the vocab and merges
        
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
        """
        assert self.count != None, "Run pretokenization before training"
        
        # Initialize with special tokens and 256 byte values
        id = 0
        vocab = {}
        for s in self.special_tokens:
            vocab[id] = s.encode("utf-8")
            id += 1
        for i in range(256):
            vocab[id] = bytes([i])
            id += 1
            
        # Calculate inital freq
        
        freq = Counter() 
        for i in self.count:
            