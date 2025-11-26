import numpy as np
from numba import jit, prange
import math
from collections import defaultdict

class ParallelTFIDF:
    """Simple Numba-accelerated TF-IDF"""
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def compute_tfidf_parallel(tf_matrix, df_array, num_docs):
        """
        Compute TF-IDF in parallel
        
        Args:
            tf_matrix: [num_docs x vocab_size] term frequency matrix
            df_array: [vocab_size] document frequency array
            num_docs: total number of documents
        
        Returns:
            tfidf_matrix: [num_docs x vocab_size]
        """
        num_docs_val, vocab_size = tf_matrix.shape
        tfidf_matrix = np.zeros((num_docs_val, vocab_size), dtype=np.float64)
        
        # Compute IDF in parallel
        idf = np.zeros(vocab_size, dtype=np.float64)
        for term_idx in prange(vocab_size):
            if df_array[term_idx] > 0:
                idf[term_idx] = math.log(num_docs / (1.0 + df_array[term_idx]))
        
        # Compute TF-IDF in parallel (over documents)
        for doc_idx in prange(num_docs_val):
            for term_idx in range(vocab_size):
                tfidf_matrix[doc_idx, term_idx] = tf_matrix[doc_idx, term_idx] * idf[term_idx]
        
        return tfidf_matrix
    
    @staticmethod
    def compute(docs: list, num_workers=4):
        """
        Compute TF-IDF scores using Numba
        
        Args:
            docs: List of document strings
            num_workers: Number of threads
        
        Returns:
            dict: Term -> average TF-IDF score
        """
        import numba
        numba.set_num_threads(num_workers)
        
        # Build vocabulary
        vocab = {}
        for doc in docs:
            for word in doc.lower().split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        
        vocab_size = len(vocab)
        num_docs = len(docs)
        
        # Build TF matrix and DF array
        tf_matrix = np.zeros((num_docs, vocab_size), dtype=np.float64)
        df_array = np.zeros(vocab_size, dtype=np.int64)
        
        for doc_idx, doc in enumerate(docs):
            words = doc.lower().split()
            word_count = len(words)
            
            # Track unique words in this doc
            unique_words = set()
            
            for word in words:
                if word in vocab:
                    word_idx = vocab[word]
                    tf_matrix[doc_idx, word_idx] += 1.0
                    unique_words.add(word_idx)
            
            # Normalize TF by document length
            if word_count > 0:
                tf_matrix[doc_idx] /= word_count
            
            # Update document frequency
            for word_idx in unique_words:
                df_array[word_idx] += 1
        
        # Compute TF-IDF (Numba parallel)
        tfidf_matrix = ParallelTFIDF.compute_tfidf_parallel(tf_matrix, df_array, num_docs)
        
        # Average TF-IDF across documents
        tfidf_scores = {}
        reverse_vocab = {idx: word for word, idx in vocab.items()}
        
        for term_idx in range(vocab_size):
            avg_score = tfidf_matrix[:, term_idx].mean()
            if avg_score > 0:
                tfidf_scores[reverse_vocab[term_idx]] = float(avg_score)
        
        return tfidf_scores