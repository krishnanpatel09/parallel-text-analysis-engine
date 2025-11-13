from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import Counter

def _count_document(doc):
    """Count words in a single document (module-level for pickling)"""
    return Counter(doc.lower().split())

class FuturesWordCount:
    """Parallel word counting using concurrent.futures"""
    
    @staticmethod
    def compute_parallel_process(docs: list, max_workers=None) -> Counter:
        """Parallel word frequency using ProcessPoolExecutor"""
        total_counter = Counter()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_count_document, doc) for doc in docs]
            
            for future in as_completed(futures):
                total_counter.update(future.result())
        
        return total_counter
    
    @staticmethod
    def compute_parallel_thread(docs: list, max_workers=None) -> Counter:
        """Parallel word frequency using ThreadPoolExecutor"""
        total_counter = Counter()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_count_document, doc) for doc in docs]
            
            for future in as_completed(futures):
                total_counter.update(future.result())
        
        return total_counter