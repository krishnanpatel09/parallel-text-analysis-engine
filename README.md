<div align="center">
  <h1>Parallel Text Analysis Engine</h1>
  <p>
    A modular Python system for large scale text processing, TF-IDF analysis, benchmarking, and Numba-based parallel acceleration.
  </p>
 <p> <a href="https://www.python.org/" target="_blank"> <img src="https://img.shields.io/badge/Python-3-blue" /> </a> <a href="https://numpy.org/" target="_blank"> <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-informational" /> </a> <a href="https://www.nltk.org/" target="_blank"> <img src="https://img.shields.io/badge/NLTK-Text%20Processing-success" /> </a> <a href="https://numba.pydata.org/" target="_blank"> <img src="https://img.shields.io/badge/Numba-Parallel%20Acceleration-orange" /> </a> <a href="https://matplotlib.org/" target="_blank"> <img src="https://img.shields.io/badge/Matplotlib-Visualization-red" /> </a> </p>
</div>

<hr>

<h2>📌 Overview</h2>
<p>
  This project implements a modular text processing engine designed to analyze large scale text corpora efficiently.
  It provides both a sequential baseline and a Numba-accelerated parallel implementation to evaluate performance
  improvements in practical text processing workloads.
</p>
<p>
  The engine processes multiple documents, performs preprocessing, computes word frequencies, calculates TF-IDF scores,
  and exports benchmarking results and visualizations. It is built to study execution time, throughput, speedup,
  parallel efficiency, and scaling behavior in a clear and repeatable way.
</p>

<h2>🎯 Project Motivation</h2>
<p>
  Modern software systems generate large volumes of unstructured text, including logs, user interactions, and application data.
  Processing this text efficiently is important when dataset size and vocabulary complexity grow.
</p>
<p>This project focuses on:</p>
<ul>
  <li>Identifying bottlenecks in text processing workflows</li>
  <li>Evaluating practical parallelization strategies</li>
  <li>Understanding trade-offs in speedup and efficiency</li>
  <li>Applying high performance computing ideas to a real software project</li>
</ul>

<h2>🚀 Key Features</h2>
<ul>
  <li>End to end text processing pipeline</li>
  <li>Synthetic dataset generation at multiple scales</li>
  <li>Word frequency computation</li>
  <li>TF-IDF calculation</li>
  <li>Sequential and parallel execution modes</li>
  <li>Numba-based acceleration using JIT compilation</li>
  <li>Benchmarking and performance visualization</li>
  <li>Export of results as JSON files and plots</li>
</ul>

<h2>🏗️ System Architecture</h2>
<p>The engine follows a modular pipeline design:</p>

<table>
  <tr>
    <td align="center"><b>1</b></td>
    <td>Dataset Generation</td>
  </tr>
  <tr>
    <td align="center"><b>2</b></td>
    <td>File Loading</td>
  </tr>
  <tr>
    <td align="center"><b>3</b></td>
    <td>Preprocessing and Tokenization</td>
  </tr>
  <tr>
    <td align="center"><b>4</b></td>
    <td>Word Frequency Computation</td>
  </tr>
  <tr>
    <td align="center"><b>5</b></td>
    <td>TF-IDF Computation</td>
  </tr>
  <tr>
    <td align="center"><b>6</b></td>
    <td>Benchmarking</td>
  </tr>
  <tr>
    <td align="center"><b>7</b></td>
    <td>Visualization</td>
  </tr>
</table>

<p>Main workflow entry point:</p>
<p><code>src/main.py</code></p>

<h2>⚙️ Execution Modes</h2>

<h3>Sequential Baseline</h3>
<ul>
  <li>Pure Python implementation</li>
  <li>Used as the reference for performance comparison</li>
  <li>Processes text stage by stage</li>
  <li>Records detailed execution metrics</li>
</ul>

<h3>Parallel Mode, Numba Accelerated</h3>
<ul>
  <li>Uses JIT compiled kernels</li>
  <li>Converts text into numeric arrays for efficient processing</li>
  <li>Applies parallel loops using <code>prange</code></li>
  <li>Accelerates core numerical computations</li>
</ul>

<h2>📂 Dataset</h2>
<p>Synthetic datasets are generated automatically for controlled and repeatable experiments.</p>

<table>
  <tr>
    <th>Dataset Size</th>
    <th>Files</th>
    <th>Purpose</th>
  </tr>
  <tr>
    <td>Small</td>
    <td>10</td>
    <td>Quick baseline and low overhead testing</td>
  </tr>
  <tr>
    <td>Medium</td>
    <td>100</td>
    <td>Balanced workload for performance comparison</td>
  </tr>
  <tr>
    <td>Large</td>
    <td>500</td>
    <td>Stress testing and scalability analysis</td>
  </tr>
</table>

<p>Dataset location:</p>
<p><code>datasets/test/</code></p>

<p>Each dataset includes:</p>
<ul>
  <li>Controlled vocabulary</li>
  <li>Consistent token distribution</li>
  <li>Predictable scaling characteristics</li>
</ul>

<h2>🧰 Tech Stack</h2>
<table>
  <tr>
    <th>Category</th>
    <th>Tools</th>
  </tr>
  <tr>
    <td>Language</td>
    <td>Python 3</td>
  </tr>
  <tr>
    <td>Numerical Processing</td>
    <td>NumPy</td>
  </tr>
  <tr>
    <td>Text Processing</td>
    <td>NLTK</td>
  </tr>
  <tr>
    <td>Parallel Acceleration</td>
    <td>Numba</td>
  </tr>
  <tr>
    <td>Visualization</td>
    <td>Matplotlib</td>
  </tr>
</table>

<h2>📁 Project Structure</h2>
<pre><code>parallel-text-analysis-engine/
├── datasets/
├── results/
├── src/
│   ├── main.py
│   ├── utils/
│   ├── sequential/
│   └── parallel/
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
└── .gitignore</code></pre>

<h2>🛠️ Installation</h2>

<p>Clone the repository:</p>
<pre><code>git clone https://github.com/your-username/parallel-text-analysis-engine.git
cd parallel-text-analysis-engine</code></pre>

<p>Install dependencies:</p>
<pre><code>pip install -r requirements.txt</code></pre>

<h2>▶️ Usage</h2>

<p>Run the full pipeline:</p>
<pre><code>python3 src/main.py</code></pre>

<p>This will:</p>
<ul>
  <li>Generate datasets</li>
  <li>Run sequential analysis</li>
  <li>Run parallel analysis</li>
  <li>Store results and plots</li>
</ul>

<h2>📊 Output</h2>

<p>All generated results are stored in:</p>
<p><code>results/</code></p>

<p>Includes:</p>
<ul>
  <li>JSON result files</li>
  <li>Benchmark comparison outputs</li>
  <li>Performance plots and charts</li>
</ul>

<p>Main output files:</p>
<pre><code>sequential_baseline.json
parallel_comprehensive.json</code></pre>

<h2>📈 Performance Highlights</h2>
<ul>
  <li>Execution time scales close to linearly with dataset size</li>
  <li>Word counting is the main computational bottleneck</li>
  <li>Numba achieves about 2x to 5x speedup depending on workload size</li>
  <li>Throughput increases significantly on larger datasets</li>
  <li>Parallel efficiency drops when worker count becomes too high</li>
  <li>Memory bandwidth limits scalability at larger sizes</li>
</ul>

<table>
  <tr>
    <th>Dataset</th>
    <th>Approximate Best Speedup</th>
  </tr>
  <tr>
    <td>Small</td>
    <td>~5.2x</td>
  </tr>
  <tr>
    <td>Medium</td>
    <td>~2.3x</td>
  </tr>
  <tr>
    <td>Large</td>
    <td>~2.5x</td>
  </tr>
</table>



<h2>📚 Key Learnings</h2>
<ul>
  <li>Data layout matters as much as the choice of tool</li>
  <li>Converting text to numeric arrays enables effective parallelism</li>
  <li>More threads do not always improve performance</li>
  <li>JIT overhead can dominate small workloads</li>
  <li>A strong sequential baseline is essential for meaningful comparison</li>
</ul>

<h2>⚠️ Limitations</h2>
<ul>
  <li>Uses synthetic datasets only</li>
  <li>Limited preprocessing features</li>
  <li>Only CPU-based parallelism is explored</li>
  <li>Some stages of the pipeline remain sequential</li>
</ul>

<h2>🔮 Future Work</h2>
<ul>
  <li>Use real world datasets</li>
  <li>Add richer NLP preprocessing</li>
  <li>Integrate GPU acceleration</li>
  <li>Explore distributed systems such as Spark</li>
  <li>Improve benchmarking with repeated trials and statistical summaries</li>
</ul>

<h2>💡 Why This Project Matters</h2>
<p>
  This project demonstrates how high performance computing concepts can be applied to a practical software problem.
  It moves beyond a simple text analytics script and builds a structured experimental platform for profiling,
  optimization, and scalability analysis.
</p>

<h2>👤 Author</h2>
<p>
  Krishna Nikunjkumar Patel
</p>

<div align="center">
  <h3>If you found this project useful, consider giving it a star.</h3>
</div>
