import numba
from numba import jit, prange
import numpy as np
from collections import Counter

class NumbaWordCount:
    """Numba-accelerated parallel word counting"""
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def count_tokens_parallel(token_ids, vocab_size):
        """Fast parallel token counting using Numba"""
        counts = np.zeros(vocab_size, dtype=np.int64)
        
        for i in prange(len(token_ids)):
            counts[token_ids[i]] += 1
        
        return counts
    
    @staticmethod
    def compute_parallel(docs: list, num_workers=None) -> Counter:
        """Parallel word frequency using Numba (num_workers ignored, uses all cores)"""
        # Build vocabulary
        vocab = {}
        vocab_list = []
        
        for doc in docs:
            for word in doc.lower().split():
                if word not in vocab:
                    vocab[word] = len(vocab)
                    vocab_list.append(word)
        
        # Convert documents to token IDs
        token_ids = []
        for doc in docs:
            for word in doc.lower().split():
                token_ids.append(vocab[word])
        
        token_ids = np.array(token_ids, dtype=np.int64)
        
        # Parallel counting
        counts = NumbaWordCount.count_tokens_parallel(token_ids, len(vocab))
        
        # Convert back to Counter
        result = Counter()
        for word, idx in vocab.items():
            if counts[idx] > 0:
                result[word] = int(counts[idx])
        
        return result