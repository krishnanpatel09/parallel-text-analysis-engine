import os
import time
from dataclasses import dataclass
from typing import List, Dict
import json

@dataclass
class ParallelBenchmarkResult:
    """Comprehensive parallel benchmark metrics"""
    implementation: str  # e.g., "multiprocessing", "threading", "numba"
    dataset_size: str
    num_files: int
    num_workers: int
    total_tokens: int
    unique_words: int
    word_freq_time: float
    tfidf_time: float
    total_time: float
    throughput: float
    speedup: float
    efficiency: float
    scalability_index: float
    
    def to_dict(self):
        return self.__dict__

class ParallelBenchmarkSuite:
    """Comprehensive benchmarking suite"""
    
    def __init__(self, sequential_results):
        self.sequential_results = {r.dataset_size: r for r in sequential_results}
        self.parallel_results = []
    
    def benchmark_implementation(self, impl_name, word_count_func, tfidf_func, 
                                docs, dataset_size, num_workers_list=[1, 2, 4, 8]):
        """Benchmark a parallel implementation with different worker counts"""
        
        for num_workers in num_workers_list:
            # Word frequency benchmark
            start = time.time()
            word_freqs = word_count_func(docs, num_workers)
            word_freq_time = time.time() - start
            
            # TF-IDF benchmark
            start = time.time()
            tfidf_scores = tfidf_func(docs, num_workers)
            tfidf_time = time.time() - start
            
            total_time = word_freq_time + tfidf_time
            total_tokens = sum(len(doc.split()) for doc in docs)
            
            # Calculate speedup
            seq_time = self.sequential_results[dataset_size].total_time
            speedup = seq_time / total_time
            efficiency = (speedup / num_workers) * 100
            
            # Scalability index (Amdahl's law approximation)
            scalability_index = speedup / num_workers
            
            result = ParallelBenchmarkResult(
                implementation=impl_name,
                dataset_size=dataset_size,
                num_files=len(docs),
                num_workers=num_workers,
                total_tokens=total_tokens,
                unique_words=len(word_freqs),
                word_freq_time=word_freq_time,
                tfidf_time=tfidf_time,
                total_time=total_time,
                throughput=total_tokens / total_time,
                speedup=speedup,
                efficiency=efficiency,
                scalability_index=scalability_index
            )
            
            self.parallel_results.append(result)
            
            print(f"✓ {impl_name} ({num_workers} workers): {total_time:.4f}s, Speedup: {speedup:.2f}x")

    def save_results(self, filename="results/parallel_comprehensive.json"):
        """Save all results to JSON safely"""
        data = {
            "sequential": [
                r.to_dict() if hasattr(r, "to_dict") else r.__dict__
                for r in self.sequential_results.values()
            ],
            "parallel": [
                r.to_dict() if hasattr(r, "to_dict") else r.__dict__
                for r in self.parallel_results
                if hasattr(r, "implementation")  # keep only real parallel results
            ]
        }

        os.makedirs("results", exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Parallel benchmark results saved to {filename}")