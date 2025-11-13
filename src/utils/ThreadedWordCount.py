from threading import Thread, Lock
from collections import Counter
from queue import Queue

class ThreadedWordCount:
    """Thread-based parallel word counting"""
    
    def __init__(self, num_threads=4):
        self.num_threads = num_threads
        self.lock = Lock()
        self.global_counter = Counter()
    
    def count_worker(self, doc_queue):
        """Worker thread for counting words"""
        local_counter = Counter()
        
        while True:
            try:
                doc = doc_queue.get(timeout=0.1)
                if doc is None:  # Sentinel value
                    break
                tokens = doc.lower().split()
                local_counter.update(tokens)
                doc_queue.task_done()
            except:
                break
        
        # Merge with global counter
        with self.lock:
            self.global_counter.update(local_counter)
    
    def compute_parallel(self, docs: list) -> Counter:
        """Parallel word frequency using threads"""
        # Reset global counter
        self.global_counter = Counter()
        
        doc_queue = Queue()
        
        # Add documents to queue
        for doc in docs:
            doc_queue.put(doc)
        
        # Add sentinel values
        for _ in range(self.num_threads):
            doc_queue.put(None)
        
        # Create and start threads
        threads = []
        for _ in range(self.num_threads):
            t = Thread(target=self.count_worker, args=(doc_queue,))
            t.start()
            threads.append(t)
        
        # Wait for completion
        for t in threads:
            t.join()
        
        return self.global_counter