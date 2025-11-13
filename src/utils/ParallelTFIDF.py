from multiprocessing import Pool, cpu_count
import math
from collections import defaultdict, Counter
import numpy as np
from numba import jit, prange

# -------------------------------
# 🧠 Top-level worker functions
# -------------------------------

def compute_df_chunk(docs_chunk):
    """Document frequency computation (top-level for pickling)"""
    df = defaultdict(int)
    for doc in docs_chunk:
        unique_terms = set(doc.lower().split())
        for term in unique_terms:
            df[term] += 1
    return df

def compute_tfidf_chunk(docs_chunk, idf):
    """TF-IDF computation (top-level for pickling)"""
    tfidf = defaultdict(float)
    for doc in docs_chunk:
        tf = Counter(doc.lower().split())
        total_terms = len(doc.lower().split())
        for term, count in tf.items():
            tf_score = count / total_terms if total_terms > 0 else 0
            tfidf[term] += tf_score * idf.get(term, 0)
    return tfidf


# -------------------------------
# 🧩 Parallel TF-IDF Class
# -------------------------------

class ParallelTFIDF:
    """Multiple parallel TF-IDF implementations"""

    @staticmethod
    def compute_multiprocessing(docs: list, num_workers=None):
        """Multiprocessing-based TF-IDF"""
        if num_workers is None:
            num_workers = cpu_count()

        # Split docs into chunks
        chunk_size = max(1, len(docs) // num_workers)
        chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]

        # ---- Parallel DF computation ----
        with Pool(num_workers) as pool:
            df_results = pool.map(compute_df_chunk, chunks)

        df = defaultdict(int)
        for result in df_results:
            for term, count in result.items():
                df[term] += count

        # ---- Compute IDF ----
        n_docs = len(docs)
        idf = {term: math.log(n_docs / (1 + count)) for term, count in df.items()}

        # ---- Parallel TF-IDF computation ----
        with Pool(num_workers) as pool:
            tfidf_results = pool.starmap(compute_tfidf_chunk, [(chunk, idf) for chunk in chunks])

        # Merge results
        tfidf_scores = defaultdict(float)
        for result in tfidf_results:
            for term, score in result.items():
                tfidf_scores[term] += score

        # Average across docs
        for term in tfidf_scores:
            tfidf_scores[term] /= n_docs

        return dict(sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True))

    # -------------------------------
    # 🧠 Numba-based OpenMP acceleration
    # -------------------------------
    @staticmethod
    @jit(nopython=True, parallel=True)
    def compute_tf_numba(token_matrix, vocab_size):
        """Numba-accelerated TF computation"""
        n_docs = token_matrix.shape[0]
        tf_matrix = np.zeros((n_docs, vocab_size), dtype=np.float64)

        for doc_id in prange(n_docs):  # OpenMP parallel loop
            doc_tokens = token_matrix[doc_id]
            doc_length = np.sum(doc_tokens > -1)  # -1 = padding
            for token_id in doc_tokens:
                if token_id >= 0:
                    tf_matrix[doc_id, token_id] += 1.0
            if doc_length > 0:
                tf_matrix[doc_id] /= doc_length
        return tf_matrix
