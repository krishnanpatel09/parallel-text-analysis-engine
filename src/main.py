import os
import sys
import time
import shutil
import argparse
from pathlib import Path
from utils.dataSetGenrator import DatasetGenerator
from utils.file_loader import load_text_files
from sequential.word_count_seq import compute_word_frequencies
from utils.TFIDF_Impl import TFIDFComputer
from utils.benchMarkResults import Benchmarker, BenchmarkResults
from utils.resultVisualiser import ResultsVisualizer
from utils.logger import log_header, log_info, log_success, log_warning

def run_sequential_baseline():
    """Run sequential baseline analysis"""
    log_header("Sequential Baseline — Multiple Dataset Sizes")
    
    # Generate datasets
    log_info("Generating datasets...")
    DatasetGenerator.create_datasets()
    
    dataset_sizes = ['small', 'medium', 'large']
    results = []
    
    for size in dataset_sizes:
        folder = Path(__file__).parent /"datasets"/"test"/size
        log_header(f"Processing {size.upper()} dataset")
        
        # Load documents
        docs = load_text_files(str(folder))
        num_files = len(docs)
        
        if not docs:
            log_info(f"No documents found in {folder}")
            continue
        
        # Count total tokens
        total_tokens = sum(len(content.split()) for content in docs)
        
        # Benchmark word frequency
        start_time = time.time()
        word_freqs = compute_word_frequencies(docs)
        word_freq_time = time.time() - start_time
        
        # Get unique words count
        unique_words = len(word_freqs)
        
        # Benchmark TF-IDF
        start_time = time.time()
        tfidf_scores = TFIDFComputer.compute_tfidf(docs, num_files)
        tfidf_time = time.time() - start_time
        
        # Record results
        benchmark = Benchmarker.measure(
            dataset_size=size,
            num_files=num_files,
            total_tokens=total_tokens,
            unique_words=unique_words,
            word_freq_time=word_freq_time,
            tfidf_time=tfidf_time
        )
        results.append(benchmark)
        
        log_success(f"✓ {size}: {benchmark.total_time:.4f}s")
        
        log_info("Top 10 frequent words:")
        for word, count in word_freqs.most_common(10):
            print(f"  {word}: {count}")
        
        log_info("Top 10 TF-IDF terms:")
        tfidf_list = list(tfidf_scores.items())[:10]
        for term, score in tfidf_list:
            print(f"  {term}: {score:.6f}")
    
    # Summary
    log_header("Sequential Baseline Summary")
    for result in results:
        print(result)
    
    # Generate visualizations
    ResultsVisualizer.generate_all_visualizations(results)
    
    return results


def run_parallel_analysis(sequential_results=None):
    """Run parallel analysis with Numba implementation"""
    log_header("Parallel Implementation — Numba Hybrid")
    
    try:
        from utils.NumbaWordCount import NumbaWordCountHybrid
        from utils.ParallelTFIDF import ParallelTFIDF
        from utils.ParallelBenchmarkResult import ParallelBenchmarkSuite
        from utils.AdvancedVisualizations import AdvancedVisualizations
    except ImportError as e:
        log_warning(f"Parallel implementations not found: {e}")
        return []
    
    # Load sequential results if not provided
    if sequential_results is None:
        try:
            import json
            results_path = Path(__file__).parent / "results" / "sequential_baseline.json"
            with open(results_path, 'r') as f:
                seq_data = json.load(f)
            
            sequential_results = []
            for result in seq_data.get('results', []):
                sequential_results.append(BenchmarkResults(
                    dataset_size=result['dataset_size'],
                    num_files=result['num_files'],
                    total_tokens=result['total_tokens'],
                    unique_words=result['unique_words'],
                    execution_time=result['total_time'],
                    throughput=result['throughput'],
                    avg_tokens_per_file=result['avg_tokens_per_file'],
                    word_freq_time=result['word_freq_time'],
                    tfidf_time=result['tfidf_time'],
                    total_time=result['total_time']
                ))
            log_info(f"✓ Loaded {len(sequential_results)} sequential baseline results")
        except FileNotFoundError:
            log_warning("Sequential baseline results not found. Running sequential baseline first...")
            sequential_results = run_sequential_baseline()
    
    # Initialize benchmark suite
    benchmark_suite = ParallelBenchmarkSuite(sequential_results)
    
    dataset_sizes = ['small', 'medium', 'large']
    
    for size in dataset_sizes:
        log_header(f"Processing {size.upper()} Dataset - Parallel")
        
        try:
            # ✅ SAME PATH as sequential
            folder = Path(__file__).parent / "datasets" / "test" / size
            docs = load_text_files(str(folder))
            
            if not docs:
                log_warning(f"No documents found in {folder}")
                continue
            
            log_info(f"✓ Loaded {len(docs)} documents")
            
            # Benchmark Numba implementation
            print("\n🚀 Numba Hybrid Implementation:")
            benchmark_suite.benchmark_implementation(
                "numba_hybrid",
                NumbaWordCountHybrid.compute_parallel,
                ParallelTFIDF.compute,
                docs, size
            )
                
        except Exception as e:
            log_warning(f"Error processing {size} dataset: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    try:
        benchmark_suite.save_results()
        log_success("Parallel results saved successfully")
    except Exception as e:
        log_warning(f"Error saving results: {e}")
    
    # Generate visualizations
    log_info("Generating visualizations...")
    try:
        for size in dataset_sizes:
            AdvancedVisualizations.plot_speedup_curves(benchmark_suite.parallel_results, size)
            AdvancedVisualizations.plot_efficiency(benchmark_suite.parallel_results, size)
        
        AdvancedVisualizations.plot_strong_scaling(benchmark_suite.parallel_results, 'numba_hybrid')
        
        log_success("Visualizations generated successfully")
    except Exception as e:
        log_warning(f"Error generating visualizations: {e}")
    
    return benchmark_suite.parallel_results


def cleanup_datasets():
    """Clean up generated test datasets"""
    log_info("Cleaning up generated test datasets...")
    time.sleep(1)
    
    test_folder = Path(__file__).parent / "datasets" / "test"
    
    if test_folder.exists():
        try:
            shutil.rmtree(test_folder)
            log_success("Test datasets deleted successfully")
        except OSError as e:
            log_warning(f"Could not delete datasets: {e}")
    else:
        log_info("No test datasets to clean up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Text Analysis Engine - Sequential vs Numba Parallel'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['sequential', 'parallel', 'both'],
        default='both',
        help='Execution mode (default: both)'
    )
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Keep datasets after execution'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("TEXT ANALYSIS ENGINE - NUMBA PARALLEL IMPLEMENTATION")
    print("="*70)
    
    sequential_results = None
    parallel_results = None
    
    try:
        if args.mode in ['sequential', 'both']:
            sequential_results = run_sequential_baseline()
        
        if args.mode in ['parallel', 'both']:
            parallel_results = run_parallel_analysis(sequential_results)
        
        print("\n" + "="*70)
        log_success("Analysis Complete!")
        print("="*70)
        
        if sequential_results:
            log_info(f"Sequential: {len(sequential_results)} datasets analyzed")
        if parallel_results:
            log_info(f"Parallel: {len(parallel_results)} benchmarks completed")
        
        log_info("Check 'results/' folder for visualizations")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        log_warning(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if not args.no_cleanup:
            cleanup_datasets()
        else:
            log_info("Datasets preserved (--no-cleanup)")