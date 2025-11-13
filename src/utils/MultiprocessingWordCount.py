from multiprocessing import Pool, cpu_count
from collections import Counter

def _count_words_chunk(docs_chunk):
    """Count words in a chunk of documents (module-level for pickling)"""
    counter = Counter()
    for doc in docs_chunk:
        tokens = doc.lower().split()
        counter.update(tokens)
    return counter

class MultiprocessingWordCount:
    """Process-based parallel word counting"""
    
    @staticmethod
    def compute_parallel(docs: list, num_workers=None) -> Counter:
        """Parallel word frequency using multiprocessing"""
        if num_workers is None:
            num_workers = cpu_count()
        
        # Split documents into chunks
        chunk_size = max(1, len(docs) // num_workers)
        chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]
        
        # Parallel processing
        with Pool(num_workers) as pool:
            results = pool.map(_count_words_chunk, chunks)
        
        # Merge results
        total_counter = Counter()
        for result in results:
            total_counter.update(result)
        
        return total_counter