import matplotlib.pyplot as plt
import numpy as np
import os

class AdvancedVisualizations:
    """Simple and clear visualizations for sequential vs parallel comparison"""
    
    @staticmethod
    def plot_execution_time_comparison(sequential_results, parallel_results):
        """
        Simple execution time comparison: Sequential vs Numba
        Shows actual execution times (not speedup)
        """
        dataset_sizes = ['small', 'medium', 'large']
        
        # Get sequential times
        seq_times = {}
        for r in sequential_results:
            seq_times[r.dataset_size] = r.total_time
        
        # Get parallel times grouped by worker count
        worker_counts = sorted(set(r.num_workers for r in parallel_results))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, dataset_size in enumerate(dataset_sizes):
            ax = axes[idx]
            
            # Get sequential time
            seq_time = seq_times.get(dataset_size, 0)
            
            # Get parallel times for this dataset
            parallel_times = []
            for num_workers in worker_counts:
                results = [r for r in parallel_results 
                          if r.dataset_size == dataset_size and r.num_workers == num_workers]
                if results:
                    parallel_times.append(results[0].total_time)
                else:
                    parallel_times.append(0)
            
            # Plot
            x_positions = [0] + list(range(1, len(worker_counts) + 1))
            times = [seq_time] + parallel_times
            labels = ['Sequential'] + [f'{w} workers' for w in worker_counts]
            
            colors = ['#e74c3c'] + ['#3498db'] * len(worker_counts)
            bars = ax.bar(x_positions, times, color=colors, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{time:.4f}s',
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            ax.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
            ax.set_title(f'{dataset_size.upper()} Dataset', fontsize=12, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        fig.suptitle('Execution Time: Sequential vs Numba Parallel', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        os.makedirs('results', exist_ok=True)
        plt.tight_layout()
        plt.savefig('results/execution_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved execution_time_comparison.png")
    
    @staticmethod
    def plot_speedup_curves(sequential_results, parallel_results, dataset_size):
        """Plot speedup curves with sequential baseline shown"""
        # Filter results for this dataset
        parallel_data = [r for r in parallel_results if r.dataset_size == dataset_size]
        seq_data = [r for r in sequential_results if r.dataset_size == dataset_size]
        
        if not parallel_data or not seq_data:
            print(f"No data for {dataset_size} dataset")
            return
        
        seq_time = seq_data[0].total_time
        
        # Sort by workers
        parallel_data.sort(key=lambda x: x.num_workers)
        
        workers = [r.num_workers for r in parallel_data]
        times = [r.total_time for r in parallel_data]
        speedups = [r.speedup for r in parallel_data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Execution Times
        ax1.axhline(y=seq_time, color='#e74c3c', linestyle='--', linewidth=2, 
                   label=f'Sequential ({seq_time:.4f}s)', zorder=1)
        ax1.plot(workers, times, marker='o', linewidth=2, markersize=10, 
                color='#3498db', label='Numba Parallel', zorder=2)
        
        for w, t in zip(workers, times):
            ax1.text(w, t, f'{t:.4f}s', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xlabel('Number of Workers', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Execution Time - {dataset_size.upper()}', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup
        max_workers = max(workers)
        ax2.plot([1, max_workers], [1, max_workers], 'k--', label='Ideal (Linear)', alpha=0.5)
        ax2.plot(workers, speedups, marker='o', linewidth=2, markersize=10, 
                color='#27ae60', label='Numba Speedup')
        
        for w, s in zip(workers, speedups):
            ax2.text(w, s, f'{s:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xlabel('Number of Workers', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Speedup', fontsize=11, fontweight='bold')
        ax2.set_title(f'Speedup vs Sequential - {dataset_size.upper()}', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        os.makedirs('results', exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'results/speedup_curves_{dataset_size}.png', dpi=300)
        plt.close()
        print(f"  ✓ Saved speedup_curves_{dataset_size}.png")
    
    @staticmethod
    def plot_efficiency(sequential_results, parallel_results, dataset_size):
        """Plot efficiency chart"""
        parallel_data = [r for r in parallel_results if r.dataset_size == dataset_size]
        
        if not parallel_data:
            return
        
        parallel_data.sort(key=lambda x: x.num_workers)
        
        workers = [r.num_workers for r in parallel_data]
        efficiencies = [r.efficiency for r in parallel_data]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(workers, efficiencies, color='#9b59b6', edgecolor='black', linewidth=1.5)
        ax.axhline(y=100, color='r', linestyle='--', linewidth=2, label='100% Efficiency')
        
        # Add value labels
        for bar, eff in zip(bars, efficiencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{eff:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Number of Workers', fontsize=11, fontweight='bold')
        ax.set_ylabel('Parallel Efficiency (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Parallel Efficiency - {dataset_size.upper()} Dataset', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(workers)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        os.makedirs('results', exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'results/efficiency_{dataset_size}.png', dpi=300)
        plt.close()
        print(f"  ✓ Saved efficiency_{dataset_size}.png")
    
    @staticmethod
    def plot_strong_scaling(sequential_results, parallel_results, implementation):
        """Plot strong scaling with sequential baseline"""
        impl_results = [r for r in parallel_results if r.implementation == implementation]
        
        if not impl_results:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        dataset_sizes = ['small', 'medium', 'large']
        
        for idx, dataset_size in enumerate(dataset_sizes):
            ax = axes[idx]
            
            # Get sequential baseline
            seq_data = [r for r in sequential_results if r.dataset_size == dataset_size]
            seq_time = seq_data[0].total_time if seq_data else None
            
            # Get parallel results
            size_results = [r for r in impl_results if r.dataset_size == dataset_size]
            size_results.sort(key=lambda x: x.num_workers)
            
            if not size_results:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(f'{dataset_size.upper()}', fontweight='bold')
                continue
            
            workers = [0] + [r.num_workers for r in size_results]
            times = [seq_time] + [r.total_time for r in size_results] if seq_time else [r.total_time for r in size_results]
            
            # Plot sequential baseline
            if seq_time:
                ax.axhline(y=seq_time, color='#e74c3c', linestyle='--', linewidth=2, 
                          label=f'Sequential ({seq_time:.4f}s)', zorder=1)
            
            # Plot parallel results
            parallel_workers = [r.num_workers for r in size_results]
            parallel_times = [r.total_time for r in size_results]
            ax.plot(parallel_workers, parallel_times, marker='o', linewidth=2, 
                   markersize=8, color='#3498db', label='Numba Parallel')
            
            ax.set_xlabel('Number of Workers', fontweight='bold')
            ax.set_ylabel('Execution Time (s)', fontweight='bold')
            ax.set_title(f'{dataset_size.upper()} Dataset', fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Strong Scaling: Sequential vs {implementation}', 
                    fontsize=14, fontweight='bold')
        
        os.makedirs('results', exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'results/strong_scaling_{implementation}.png', dpi=300)
        plt.close()
        print(f"  ✓ Saved strong_scaling_{implementation}.png")