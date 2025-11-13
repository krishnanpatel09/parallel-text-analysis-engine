import matplotlib.pyplot as plt
import numpy as np

class AdvancedVisualizations:
    """Advanced visualization suite"""
    
    @staticmethod
    def plot_speedup_curves(results, dataset_size):
        """Plot speedup vs number of workers for all implementations"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        implementations = set(r.implementation for r in results if r.dataset_size == dataset_size)
        
        for impl in implementations:
            impl_results = [r for r in results if r.implementation == impl and r.dataset_size == dataset_size]
            impl_results.sort(key=lambda x: x.num_workers)
            
            workers = [r.num_workers for r in impl_results]
            speedups = [r.speedup for r in impl_results]
            
            ax.plot(workers, speedups, marker='o', linewidth=2, markersize=8, label=impl)
        
        # Ideal speedup line
        max_workers = max(r.num_workers for r in results if r.dataset_size == dataset_size)
        ax.plot([1, max_workers], [1, max_workers], 'k--', label='Ideal (Linear)', alpha=0.5)
        
        ax.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup', fontsize=12, fontweight='bold')
        ax.set_title(f'Speedup Curves - {dataset_size.upper()} Dataset', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/speedup_curves_{dataset_size}.png', dpi=300)
        plt.close()
    
    @staticmethod
    def plot_efficiency(results, dataset_size):
        """Plot parallel efficiency"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        implementations = set(r.implementation for r in results if r.dataset_size == dataset_size)
        
        x = np.arange(len(implementations))
        width = 0.15
        
        worker_counts = sorted(set(r.num_workers for r in results if r.dataset_size == dataset_size))
        
        for i, num_workers in enumerate(worker_counts):
            efficiencies = []
            for impl in implementations:
                impl_results = [r for r in results 
                              if r.implementation == impl 
                              and r.dataset_size == dataset_size 
                              and r.num_workers == num_workers]
                if impl_results:
                    efficiencies.append(impl_results[0].efficiency)
                else:
                    efficiencies.append(0)
            
            ax.bar(x + i*width, efficiencies, width, label=f'{num_workers} workers')
        
        ax.set_xlabel('Implementation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Efficiency (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Parallel Efficiency - {dataset_size.upper()} Dataset', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(worker_counts) - 1) / 2)
        ax.set_xticklabels(implementations, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/efficiency_{dataset_size}.png', dpi=300)
        plt.close()
    
    @staticmethod
    def plot_strong_scaling(results, implementation):
        """Plot strong scaling analysis"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        dataset_sizes = ['small', 'medium', 'large']
        
        for idx, dataset_size in enumerate(dataset_sizes):
            impl_results = [r for r in results 
                          if r.implementation == implementation 
                          and r.dataset_size == dataset_size]
            impl_results.sort(key=lambda x: x.num_workers)
            
            workers = [r.num_workers for r in impl_results]
            times = [r.total_time for r in impl_results]
            
            axes[idx].plot(workers, times, marker='o', linewidth=2, markersize=8, color='#e74c3c')
            axes[idx].set_xlabel('Number of Workers', fontweight='bold')
            axes[idx].set_ylabel('Execution Time (s)', fontweight='bold')
            axes[idx].set_title(f'{dataset_size.upper()} Dataset', fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        
        fig.suptitle(f'Strong Scaling Analysis - {implementation}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'results/strong_scaling_{implementation}.png', dpi=300)
        plt.close()