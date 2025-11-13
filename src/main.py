import os
import sys
import time
import shutil
import argparse
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
        folder = f"datasets/test/{size}"
        log_header(f"Processing {size.upper()} dataset")
        
        # Load documents
        docs = load_text_files(folder)
        num_files = len(docs)
        
        if not docs:
            log_info(f"No documents found in {folder}")
            continue
        
        # Count total tokens
        if isinstance(docs, list):
            total_tokens = sum(len(content.split()) for content in docs)
        else:
            total_tokens = sum(len(content.split()) for content in docs.values())
        
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
        
        # Record results with correct parameters
        benchmark = Benchmarker.measure(
            dataset_size=size,
            num_files=num_files,
            total_tokens=total_tokens,
            unique_words=unique_words,
            word_freq_time=word_freq_time,
            tfidf_time=tfidf_time
        )
        results.append(benchmark)
        
        log_success(f"Completed {size} dataset analysis")
        print(benchmark)
        
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
    """Run parallel analysis with multiple implementations"""
    log_header("Parallel Implementation — Comprehensive Evaluation")
    
    try:
        from utils.MultiprocessingWordCount import MultiprocessingWordCount
        from utils.ThreadedWordCount import ThreadedWordCount
        from utils.NumbaWordCount import NumbaWordCount
        from utils.FuturesWordCount import FuturesWordCount
        from utils.ParallelTFIDF import ParallelTFIDF
        from utils.ParallelBenchmarkResult import ParallelBenchmarkSuite
        from utils.AdvancedVisualizations import AdvancedVisualizations
    except ImportError as e:
        log_warning(f"Parallel implementations not found: {e}")
        log_info("Please ensure all parallel implementation files are created.")
        return []
    
    # Load sequential results if not provided
    if sequential_results is None:
        try:
            import json
            with open('results/sequential_baseline.json', 'r') as f:
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
    
    # Check if datasets exist
    if not os.path.exists('datasets/test/small'):
        log_info("Generating datasets...")
        DatasetGenerator.create_datasets()
    
    # Initialize benchmark suite
    benchmark_suite = ParallelBenchmarkSuite(sequential_results)
    
    dataset_sizes = ['small', 'medium', 'large']
    
    for size in dataset_sizes:
        log_header(f"Processing {size.upper()} Dataset - Parallel")
        
        try:
            docs = load_text_files(f"datasets/test/{size}")
            
            if not docs:
                log_warning(f"No documents found in datasets/test/{size}")
                continue
            
            log_info(f"✓ Loaded {len(docs)} documents")
            
            # Benchmark all implementations
            print("\n1. Multiprocessing Implementation:")
            try:
                benchmark_suite.benchmark_implementation(
                    "multiprocessing",
                    MultiprocessingWordCount.compute_parallel,
                    ParallelTFIDF.compute_multiprocessing,
                    docs, size
                )
            except Exception as e:
                log_warning(f"Error in multiprocessing: {e}")
            
            print("\n2. Threading Implementation:")
            try:
                benchmark_suite.benchmark_implementation(
                    "threading",
                    lambda docs, w: ThreadedWordCount(w).compute_parallel(docs),
                    ParallelTFIDF.compute_multiprocessing,
                    docs, size
                )
            except Exception as e:
                log_warning(f"Error in threading: {e}")
            
            print("\n3. Numba Implementation:")
            try:
                benchmark_suite.benchmark_implementation(
                    "numba",
                    NumbaWordCount.compute_parallel,
                    ParallelTFIDF.compute_multiprocessing,
                    docs, size
                )
            except Exception as e:
                log_warning(f"Error in numba: {e}")
            
            print("\n4. Concurrent.futures Implementation:")
            try:
                benchmark_suite.benchmark_implementation(
                    "futures_process",
                    FuturesWordCount.compute_parallel_process,
                    ParallelTFIDF.compute_multiprocessing,
                    docs, size
                )
            except Exception as e:
                log_warning(f"Error in futures: {e}")
                
        except Exception as e:
            log_warning(f"Error processing {size} dataset: {e}")
            continue
    
    # Save results
    try:
        benchmark_suite.save_results()
        log_success("Parallel results saved successfully")
    except Exception as e:
        log_warning(f"Error saving parallel results: {e}")
    
    # Generate visualizations
    log_info("Generating parallel visualizations...")
    try:
        for size in dataset_sizes:
            AdvancedVisualizations.plot_speedup_curves(benchmark_suite.parallel_results, size)
            AdvancedVisualizations.plot_efficiency(benchmark_suite.parallel_results, size)
        
        for impl in ['multiprocessing', 'threading', 'numba', 'futures_process']:
            AdvancedVisualizations.plot_strong_scaling(benchmark_suite.parallel_results, impl)
        
        log_success("All parallel visualizations generated successfully")
    except Exception as e:
        log_warning(f"Error generating visualizations: {e}")
    
    return benchmark_suite.parallel_results


def cleanup_datasets():
    """Clean up generated datasets"""
    log_info("Cleaning up datasets...")
    time.sleep(1)
    
    if os.path.exists("datasets"):
        try:
            shutil.rmtree("datasets")
            log_success("Datasets folder deleted successfully")
        except OSError as e:
            log_warning(f"Could not delete datasets folder: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Text Analysis Engine - Sequential and Parallel Implementation'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['sequential', 'parallel', 'both'],
        default='both',
        help='Execution mode: sequential, parallel, or both (default: both)'
    )
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Keep dataset files after execution'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("TEXT ANALYSIS ENGINE - PARALLEL IMPLEMENTATION PROJECT")
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
            log_info(f"Sequential results: {len(sequential_results)} datasets analyzed")
        if parallel_results:
            log_info(f"Parallel results: {len(parallel_results)} benchmarks completed")
        
        log_info("Check the 'results/' folder for detailed reports and visualizations")
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        log_warning(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if not args.no_cleanup:
            cleanup_datasets()
        else:
            log_info("Datasets preserved (--no-cleanup flag set)")