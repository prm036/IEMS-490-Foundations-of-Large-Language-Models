from collections import Counter

class SimpleBPE:
    """
    Implement Byte Pair Encoding (BPE) for a given corpus.
    Args:
        corpus: A list of strings representing the training corpus.
    Returns:
        The learned vocabulary and merges.
    
    This function learns a vocabulary of size target_vocab_size from the given 
    corpus using the BPE algorithm. It iteratively merges the most frequent 
    byte pairs in the corpus until the target vocabulary size is reached.  
    The learned vocabulary and merges are stored in the instance variables  
    `vocab` and `merges`, respectively.
    """
    def __init__(self):
        self.target_vocab_size = 1000
        self.vocab = {}
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}

    def get_stats(self, tokens):
        pairs = Counter()
        for word, freq in tokens.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs 

    def merge_vocab(self, pair, tokens):
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        new_tokens = {}
        for word, freq in tokens.items():
            new_word = word.replace(bigram, replacement)
            new_tokens[new_word] = freq
        return new_tokens

    def fit(self, corpus):
        # 1. Initialize tokens
        tokens = Counter([' '.join(list(word)) + ' </w>' for word in corpus])
        initial_vocab_size = len(set(''.join(corpus))) + 1
        num_merges = self.target_vocab_size - initial_vocab_size
        
        # 2. Repeatedly find and merge most frequent pair
        for _ in range(num_merges):
            pairs = self.get_stats(tokens)
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            tokens = self.merge_vocab(best, tokens)
            self.merges.append(best)
            
        # 3. Build the vocab using unique subwords, not the whole words.
        unique_subwords = set()
        for word in tokens.keys():
            # word.split() breaks 'l o w er </w>' into ['l', 'o', 'w', 'er', '</w>']
            unique_subwords.update(word.split()) 
            
        self.vocab = {tok: i for i, tok in enumerate(unique_subwords)}
        self.build_vocab_ids()

    def build_vocab_ids(self):
        self.token_to_id = {token: i for i, token in enumerate(self.vocab.keys())}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    def encode(self, word, return_ids=False):
        word = list(word) + ['</w>']
        while True:
            pairs = [(word[i], word[i+1]) for i in range(len(word) - 1)]
            mergeable = [p for p in pairs if p in self.merges]
            
            if not mergeable:
                break
                
            # Iterate through learned merges in order
            for p in self.merges:
                if p in pairs:
                    i = pairs.index(p)
                    word[i:i+2] = [''.join(p)]
                    break # Breaks the FOR loop to restart the WHILE loop
                    
        if return_ids:
            return [self.token_to_id.get(tok, -1) for tok in word] # Use -1 for "UNK"
        return word

    def decode(self, ids):
        tokens = [self.id_to_token.get(i, "") for i in ids]
        word = ''.join(tokens)
        if word.endswith('</w>'):
            word = word[:-4]
        return word