<h1>PARALLEL TEXT ANALYSIS ENGINE</h1>

Overview

This project implements a modular text processing engine designed to analyze large-scale text corpora efficiently. It provides both a sequential baseline and a parallel implementation using Numba to evaluate performance improvements in real-world text processing workflows.

The system processes multiple documents, performs preprocessing, computes word frequencies, and calculates TF-IDF scores. It also includes benchmarking and visualization to study execution time, throughput, speedup, and parallel efficiency.

Key Features

• End-to-end text processing pipeline
• Synthetic dataset generation at multiple scales
• Word frequency computation
• TF-IDF calculation
• Sequential and parallel execution modes
• Numba-based acceleration using JIT compilation
• Performance benchmarking and visualization
• Export of results as JSON and plots

Project Motivation

Modern systems generate large volumes of unstructured text such as logs and user interactions. Processing this data efficiently is critical.

This project focuses on:

• Identifying bottlenecks in text processing
• Evaluating parallelization strategies
• Understanding performance trade-offs such as speedup and efficiency
• Applying high-performance computing concepts in practice

System Architecture

The system follows a modular pipeline design:

Dataset Generation
File Loading
Preprocessing and Tokenization
Word Frequency Computation
TF-IDF Computation
Benchmarking
Visualization

The main workflow is controlled through:

src/main.py

Execution Modes
Sequential Baseline

• Pure Python implementation
• Used as reference for performance comparison
• Processes text step-by-step
• Records execution metrics

Parallel (Numba-Accelerated)

• Uses JIT-compiled kernels
• Converts text into numeric arrays
• Applies parallel loops using prange
• Improves performance of core computations

Dataset

Synthetic datasets are generated automatically.

Location:

datasets/test/

Sizes:

• Small: 10 files
• Medium: 100 files
• Large: 500 files

Each dataset includes:

• Controlled vocabulary
• Token distribution
• Predictable scaling

This ensures repeatable experiments.

Tech Stack

• Python 3
• NumPy
• NLTK
• Numba
• Matplotlib

Installation

Clone the repository:

git clone https://github.com/your-username/parallel-text-analysis-engine.git

cd parallel-text-analysis-engine

Install dependencies:

pip install -r requirements.txt

Usage

Run the full pipeline:

python3 src/main.py

This will:

• Generate datasets
• Run sequential analysis
• Run parallel analysis
• Save results and plots

Output

Results are stored in:

results/

Includes:

• JSON files

sequential_baseline.json
parallel_comprehensive.json

• Plots

execution time
throughput
speedup
efficiency
Performance Highlights

Key findings from experiments :

• Execution time scales linearly with dataset size
• Word counting is the main bottleneck
• Numba achieves 2x to 5x speedup
• Throughput increases significantly on large datasets
• Parallel efficiency decreases with too many workers
• Memory bandwidth limits scalability

Example results:

• Small dataset speedup: ~5.2x
• Medium dataset speedup: ~2.3x
• Large dataset speedup: ~2.5x

Project Structure

parallel-text-analysis-engine/
│
├── datasets/
├── results/
├── src/
│ ├── main.py
│ ├── utils/
│ ├── sequential/
│ ├── parallel/
│
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── .gitignore

Key Learnings

• Data layout matters more than tools
• Converting text to numeric arrays enables parallelism
• More threads do not always improve performance
• JIT overhead affects small workloads
• Strong baseline is critical for evaluation

Limitations

• Uses synthetic datasets only
• Limited preprocessing features
• Only CPU-based parallelism explored
• Some pipeline stages remain sequential

Future Work

• Use real-world datasets
• Add advanced NLP preprocessing
• Integrate GPU acceleration
• Explore distributed systems like Spark
• Improve benchmarking with multiple runs
