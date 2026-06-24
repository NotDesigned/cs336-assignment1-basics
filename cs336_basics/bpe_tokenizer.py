from typing import Iterable

from cs336_basics.pretokenization import pretokenize, text_pretokenize
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
        self.special_tokens = sorted(set(special_tokens), reverse=True) if special_tokens is not None else None
        self.reverse_vocab = {v:k for k,v in vocab.items()}

    def train_from_file(self, file_path:str, vocab_size: int, special_tokens: list[str]):
        print("Pretokenizing...")
        self.pretokenize(file_path, special_tokens)
        print("End of pretokenization, start training...")
        vocab, merges = self.train(vocab_size, special_tokens)
        self.from_vocab_merges(vocab, merges, special_tokens)
    
    def pretokenize(self, file_path:str, special_tokens: list[str] | None):
        self.count = pretokenize(file_path, special_tokens=special_tokens)
     
    def train(self, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """v
        Train the Tokenizer and return the vocab and merges
        """
        
        # Initialize with special tokens and 256 byte values
        id = 0
        vocab:dict[int, bytes] = {}
        for i in range(256):
            vocab[id] = bytes([i])
            id += 1
        for s in special_tokens:
            vocab[id] = s.encode("utf-8")
            id += 1
        merges: list[tuple[bytes, bytes]] = []
        # Calculate inital freq and current expression
        
        freq = Counter()
        current_expression: dict[str, list[bytes]] = dict()
        appear: dict[tuple[bytes,bytes], set[str]] = {}
        for pretokens in self.count:
            count = self.count[pretokens]
            _by = pretokens.encode("utf-8")
            
            # Build global counter for each vocal                
            for i in range(len(_by)-1):
                freq[(bytes([_by[i]]),bytes([_by[i+1]]))] += count
                if not (bytes([_by[i]]),bytes([_by[i+1]])) in appear:
                    appear[(bytes([_by[i]]),bytes([_by[i+1]]))] = set()
                appear[(bytes([_by[i]]),bytes([_by[i+1]]))].add(pretokens)

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
            # for pretokens, symbol_sequence in current_expression.items():
            tmp = appear[(target[0],target[1])].copy()
            for pretokens in tmp:
                symbol_sequence = current_expression[pretokens]
                newsymbol_sequence = []
                pretoken_count = self.count[pretokens]

                for i in range(len(symbol_sequence)-1):
                    freq[(symbol_sequence[i], symbol_sequence[i+1])] -= pretoken_count
                    if not (symbol_sequence[i], symbol_sequence[i+1]) in appear:
                        continue 
                    if pretokens in appear[(symbol_sequence[i],symbol_sequence[i+1])]:
                        appear[((symbol_sequence[i],symbol_sequence[i+1]))].remove(pretokens)
                    if len(appear[(symbol_sequence[i],symbol_sequence[i+1])]) == 0:
                        del appear[(symbol_sequence[i],symbol_sequence[i+1])]
                    
                i = 0 
                while i < len(symbol_sequence):
                    if i+1 < len(symbol_sequence):
                        if symbol_sequence[i] == target[0] and symbol_sequence[i+1] == target[1]:
                            newsymbol_sequence.append(merged)
                            i += 2
                        else:
                            newsymbol_sequence.append(symbol_sequence[i])
                            i += 1
                    else:
                        newsymbol_sequence.append(symbol_sequence[i])
                        i += 1
                
                for i in range(len(newsymbol_sequence)-1):
                    if not (newsymbol_sequence[i],newsymbol_sequence[i+1]) in appear:
                        appear[(newsymbol_sequence[i],newsymbol_sequence[i+1])] = set()
                    appear[(newsymbol_sequence[i],newsymbol_sequence[i+1])].add(pretokens)
                    freq[(newsymbol_sequence[i], newsymbol_sequence[i+1])] += pretoken_count
                        
                current_expression[pretokens] = newsymbol_sequence
            
            freq += Counter() # remove 0 
             
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
                        i += 2
                    else:
                        new_expression.append(expression[i])
                        i += 1
                expression = pretoken_expression[pretoken] = new_expression
    
        for pretoken in pretokenized_text:
            for token in pretoken_expression[pretoken]:
                ret.append(self.reverse_vocab[token])
        
        return ret
    
    def pretoken_iterable(self, f:Iterable[str]):
        """stream through the iterable and yield pretoken

        Args:
            f (Iterable[str]): string stream
        """
        # The idea is that we maintain a buffer
        # The key is how can we tell that whether a byte in the unsafe buffer become safe? If it cannot be any merging prefix. It must be safe.
        # After appending a chunk, pretokenize the current buffer.
        # Encode/yield every pretoken except the last one.
        # Keep the exact text of the last pretoken in buffer.
        # At EOF, encode/yield the remaining buffer.
        buffer = ""
        for string in f:
            buffer += string
            max_special_token_start = len(buffer)
            while max_special_token_start > 0 and self.special_tokens:
                flag = True
                for special_token in self.special_tokens:
                    if special_token.startswith(buffer[max_special_token_start-1:]):
                       max_special_token_start -= 1
                       flag=False
                       break
                if flag:
                    break
            
            suffix = buffer[max_special_token_start:]
            buffer = buffer[:max_special_token_start]
            
            ret = text_pretokenize(buffer, special_tokens=self.special_tokens)
            if len(ret) > 1:
                for i in range(len(ret)-1):
                    yield ret[i]
            
            if len(ret) > 0:
                buffer = ret[-1]

            buffer += suffix

        yield buffer
                
    
    def encode_iterable(self, f: Iterable[str]):
        """stream through the iterable and yield token

        Args:
            f (Iterable[str]): string stream
        """
        pretoken_expression: dict[str, list[bytes]] = {}
        for pretoken in self.pretoken_iterable(f):
            if pretoken not in pretoken_expression:    
                pretoken_byte = pretoken.encode("utf-8")
                if self.special_tokens is not None and pretoken in self.special_tokens:
                    pretoken_expression[pretoken] = [pretoken_byte]
                else:
                    pretoken_expression[pretoken] = [pretoken_byte[i:i+1] for i in range(len(pretoken_byte))] 

            expression = pretoken_expression[pretoken]
            for merge in self.merges:
                new_expression = []
                i = 0
                while i < len(expression):
                    if i+1 < len(expression) and (expression[i], expression[i+1]) == merge:
                        new_expression.append(expression[i]+expression[i+1])
                        i += 2
                    else:
                        new_expression.append(expression[i])
                        i += 1
                expression = pretoken_expression[pretoken] = new_expression
            
            for token in pretoken_expression[pretoken]:
                yield self.reverse_vocab[token]
        
            
    
    def decode(self, tokens: list[int]) -> str:
        pieces: list[bytes] = []
        for token in tokens:
            pieces.append(self.vocab[token])
        ret: bytes = b"".join(pieces)
        return ret.decode("utf-8",errors="ignore")

if __name__ == "__main__":
    s = BPE_Tokenizer()
    s.train_from_file("data/TinyStoriesV2-GPT4-train.txt", 10000, ['<|endoftext|>'])

    with open("data/test.out", "w") as f:
        f.write(str(s.vocab))
        f.write(str(s.merges))