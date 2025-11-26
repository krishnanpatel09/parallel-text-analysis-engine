import numpy as np
from numba import jit, prange
from collections import Counter

class NumbaWordCountHybrid:
    """Simple, fast Numba word count"""
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def count_tokens_parallel(token_ids, vocab_size):
        """Fast parallel token counting"""
        counts = np.zeros(vocab_size, dtype=np.int64)
        
        # Parallel counting
        for i in prange(len(token_ids)):
            counts[token_ids[i]] += 1
        
        return counts
    
    @staticmethod
    def compute_parallel(docs: list, num_workers=4) -> Counter:
        """
        Fast parallel word counting
        
        Args:
            docs: List of document strings
            num_workers: Number of threads (controlled by numba)
        
        Returns:
            Counter with word frequencies
        """
        import numba
        numba.set_num_threads(num_workers)
        
        # Build vocabulary (fast single pass)
        vocab = {}
        token_list = []
        
        for doc in docs:
            tokens = doc.lower().split()
            for word in tokens:
                if word not in vocab:
                    vocab[word] = len(vocab)
                token_list.append(vocab[word])
        
        # Convert to numpy
        token_ids = np.array(token_list, dtype=np.int64)
        vocab_size = len(vocab)
        
        # Parallel counting with Numba
        counts = NumbaWordCountHybrid.count_tokens_parallel(token_ids, vocab_size)
        
        # Convert back to Counter
        result = Counter()
        reverse_vocab = {idx: word for word, idx in vocab.items()}
        for idx in range(vocab_size):
            if counts[idx] > 0:
                result[reverse_vocab[idx]] = int(counts[idx])
        
        return result