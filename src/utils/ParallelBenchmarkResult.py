import os
import time
from dataclasses import dataclass
import json

@dataclass
class ParallelBenchmarkResult:
    """Simple benchmark result"""
    implementation: str
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
    
    def to_dict(self):
        return self.__dict__

class ParallelBenchmarkSuite:
    """Simple benchmarking suite"""
    
    def __init__(self, sequential_results):
        self.sequential_results = {r.dataset_size: r for r in sequential_results}
        self.parallel_results = []
    
    def benchmark_implementation(self, impl_name, word_count_func, tfidf_func, 
                                docs, dataset_size, num_workers_list=[1, 2, 4, 8]):
        """Benchmark with simple timing"""
        
        for num_workers in num_workers_list:
            # Word frequency benchmark
            start = time.time()
            word_freqs = word_count_func(docs, num_workers)
            word_freq_time = time.time() - start
            
            # TF-IDF benchmark
            start = time.time()
            tfidf_scores = tfidf_func(docs, num_workers)
            tfidf_time = time.time() - start
            
            # Calculate metrics
            total_time = word_freq_time + tfidf_time
            total_tokens = sum(len(doc.split()) for doc in docs)
            
            # Get sequential baseline
            seq_time = self.sequential_results[dataset_size].total_time
            
            # Calculate speedup
            speedup = seq_time / total_time
            efficiency = (speedup / num_workers) * 100
            
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
                efficiency=efficiency
            )
            
            self.parallel_results.append(result)
            
            print(f"   ✓ {num_workers} workers: {total_time:.4f}s | Speedup: {speedup:.2f}x | Efficiency: {efficiency:.2f}%")

    def save_results(self, filename="results/parallel_comprehensive.json"):
        """Save results"""
        data = {
            "sequential": [r.to_dict() if hasattr(r, "to_dict") else r.__dict__ 
                          for r in self.sequential_results.values()],
            "parallel": [r.to_dict() for r in self.parallel_results]
        }

        os.makedirs("results", exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✅ Results saved to {filename}")