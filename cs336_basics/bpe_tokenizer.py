from .pretokenization import pretokenize, text_pretokenize
from collections import Counter

class BPE_Tokenizer:
    def __init__(self):
        self.count: Counter = Counter()
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = [] 
        self.special_tokens = []
        self.reverse_vocab: dict[bytes, int] = {}
    
    def from_vocab_merges(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.reverse_vocab = {v:k for k,v in vocab.items()}

    def train_from_file(self, file_path:str, vocab_size: int, special_tokens: list[str]):
        self.pretokenize(file_path, special_tokens)
        vocab, merges = self.train(vocab_size, special_tokens)
        self.from_vocab_merges(vocab, merges, special_tokens)
    
    def pretokenize(self, file_path:str, special_tokens: list[str] | None):
        self.count = pretokenize(file_path, special_tokens=special_tokens)
     
    def train(self, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Train the Tokenizer and return the vocab and merges
        """
        assert self.count != None, "Run pretokenization before training"
        
        # Initialize with special tokens and 256 byte values
        id = 0
        vocab:dict[int, bytes] = {}
        for s in special_tokens:
            vocab[id] = s.encode("utf-8")
            id += 1
        for i in range(256):
            vocab[id] = bytes([i])
            id += 1
        merges: list[tuple[bytes, bytes]] = []
        # Calculate inital freq and current expression
        
        freq = Counter()
        current_expression: dict[str, list[bytes]] = dict()
        for pretokens in self.count:
            count = self.count[pretokens]
            _by = pretokens.encode("utf-8")
            
            # Build global counter for each vocal                
            for i in range(len(_by)-1):
                freq[(bytes([_by[i]]),bytes([_by[i+1]]))] += count

            # Build local expression for each pretoken
            for i in range(len(_by)):
                if not pretokens in current_expression:
                    current_expression[pretokens] = []
                current_expression[pretokens].append(bytes([_by[i]]))
        
        # Merges
        while True:
            
            if len(freq) == 0:
                break
            
            target:tuple[bytes,bytes]
            
            # print("Before Merged:\n", freq)
            
            target, count = freq.most_common(1)[0]
            
            # print("Merging:", target)

            if count == 0:
                break
            
            # Merge and add to vocab
            vocab[id] = target[0] + target[1]
            merged = vocab[id]
            merges.append((target[0], target[1]))
            id += 1
            
            # Rebuild
            freq = Counter()
            for pretokens, symbol_sequence in current_expression.items():
                newsymbol_sequence = []
                i = 0 
                pretoken_count = self.count[pretokens]

                while i < len(symbol_sequence):
                    if i+1 < len(symbol_sequence) and symbol_sequence[i] == target[0] and symbol_sequence[i+1] == target[1]:
                        newsymbol_sequence.append(merged)
                        i += 2
                    else:
                        newsymbol_sequence.append(symbol_sequence[i])
                        i += 1
                        
                current_expression[pretokens] = newsymbol_sequence

                i = 0
                while i < len(newsymbol_sequence) - 1:
                    freq[(newsymbol_sequence[i], newsymbol_sequence[i+1])] += pretoken_count
                    i += 1
            
            # print(freq)
            # print(current_expression)
            
            if id == vocab_size:
                break

        return vocab, merges
    
    def encode(self, text: str) -> list[int]:
        # pretokenize the text, then byte each pretoken
        
        pretokenized_text = text_pretokenize(text, self.special_tokens)
        
        unique_pretokens = set(pretokenized_text)
        
        ret:list[int] = []
        
        # For each unique pretoken, we calculate the targeted token decomposition (merges) for it.
        pretoken_expression: dict[str, list[bytes]] = {}
        
        for pretoken in unique_pretokens:
            pretoken_byte = pretoken.encode("utf-8")
            if self.special_tokens is not None and pretoken in self.special_tokens:
                pretoken_expression[pretoken] = [pretoken_byte]
            else:
                pretoken_expression[pretoken] = [pretoken_byte[i:i+1] for i in range(len(pretoken_byte))]
        
        # Apply merges to each expression with priority
        for pretoken, expression in pretoken_expression.items():
            for merge in self.merges:
                new_expression = []
                i = 0
                while i < len(expression):
                    if i+1 < len(expression) and (expression[i], expression[i+1]) == merge:
                        new_expression.append(expression[i]+expression[i+1])
                        flag = True
                        i += 2
                    else:
                        new_expression.append(expression[i])
                        i += 1
                pretoken_expression[pretoken] = new_expression
    
        for pretoken in pretokenized_text:
            for token in pretoken_expression[pretoken]:
                ret.append(self.reverse_vocab[token])
        
        return ret
    
    def decode(self, tokens: list[int]) -> str:
        pieces: list[bytes] = []
        for token in tokens:
            pieces.append(self.vocab[token])
        ret: bytes = b"".join(pieces)
        return ret.decode("utf-8")