"""
Plot comparison between NFT, FT, and improved Mermaid results
using multiple similarity metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from metric import evaluate_directories


def plot_metric_comparison(include_improved=False):
    """Create comprehensive plots comparing NFT vs FT against groundT."""
    
    print("Evaluating NFT vs groundT...")
    nft_results = evaluate_directories("NFT", "groundT", 66)
    
    print("\nEvaluating FT vs groundT...")
    ft_results = evaluate_directories("FT", "groundT", 66)
    
    improved_results = None
    if include_improved:
        print("\nEvaluating improved Mermaid outputs vs groundT...")
        improved_results = evaluate_directories(
            "mermaid_outputs_improved", "groundT", 66)
    
    if not nft_results or not ft_results:
        print("Error: Could not get results for comparison")
        return
    
    if include_improved and not improved_results:
        print("Warning: Could not get improved results, "
              "continuing without them")
        include_improved = False
        print("Error: Could not get results for comparison")
        return
    
    # Extract metrics (excluding distance metrics)
    metrics = [key.replace('avg_', '') for key in nft_results.keys()
               if 'similarity' in key]
    
    nft_scores = [nft_results[f'avg_{metric}'] for metric in metrics]
    ft_scores = [ft_results[f'avg_{metric}'] for metric in metrics]
    
    if include_improved:
        improved_scores = [improved_results[f'avg_{metric}']
                           for metric in metrics]
    
    # Clean up metric names for display
    metric_names = [metric.replace('_similarity', '').replace('_', ' ').title()
                    for metric in metrics]
    
    # Create the comparison plot
    if include_improved:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        width = 0.25
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        width = 0.35
    
    # Plot 1: Bar chart comparison
    x = np.arange(len(metrics))
    
    if include_improved:
        bars1 = ax1.bar(x - width, nft_scores, width, label='NFT',
                        alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x, ft_scores, width, label='FT',
                        alpha=0.8, color='lightcoral')
        bars3 = ax1.bar(x + width, improved_scores, width, 
                        label='Improved', alpha=0.8, color='lightgreen')
    else:
        bars1 = ax1.bar(x - width/2, nft_scores, width, label='NFT',
                        alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, ft_scores, width, label='FT',
                        alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Similarity Metrics')
    ax1.set_ylabel('Similarity Score (%)')
    if include_improved:
        ax1.set_title('NFT vs FT vs Improved Similarity Comparison\n'
                      '(vs Ground Truth)')
    else:
        ax1.set_title('NFT vs FT Similarity Comparison\n(vs Ground Truth)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    
    if include_improved:
        bars3 = ax1.bar(x + width, improved_scores, width,
                        label='Improved', alpha=0.8, color='lightgreen')
        for bar in bars3:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
    
    nft_scores_radar = nft_scores + [nft_scores[0]]  # Complete the circle
    ft_scores_radar = ft_scores + [ft_scores[0]]
    
    if include_improved:
        improved_scores_radar = improved_scores + [improved_scores[0]]
    
    ax2 = plt.subplot(122, projection='polar')
    ax2.plot(angles, nft_scores_radar, 'o-', linewidth=2,
             label='NFT', color='skyblue')
    ax2.fill(angles, nft_scores_radar, alpha=0.25, color='skyblue')
    
    ax2.plot(angles, ft_scores_radar, 'o-', linewidth=2,
             label='FT', color='lightcoral')
    ax2.fill(angles, ft_scores_radar, alpha=0.25, color='lightcoral')
    
    if include_improved:
        ax2.plot(angles, improved_scores_radar, 'o-', linewidth=2,
                 label='Improved', color='lightgreen')
        ax2.fill(angles, improved_scores_radar, alpha=0.25,
                 color='lightgreen')
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metric_names)
    ax2.set_ylim(0, 100)
    if include_improved:
        ax2.set_title('Similarity Metrics Radar Chart\n'
                      '(NFT vs FT vs Improved vs Ground Truth)')
    else:
        ax2.set_title('Similarity Metrics Radar Chart\n'
                      '(NFT vs FT vs Ground Truth)')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    ax2.grid(True)
    
    plt.tight_layout()
    filename = ('comparison_with_improved.png' if include_improved
                else 'nft_vs_ft_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("DETAILED COMPARISON RESULTS")
    print("="*70)
    if include_improved:
        print(f"{'Metric':<20} {'NFT':<12} {'FT':<12} {'Improved':<12} "
              f"{'Best':<8}")
        print("-"*70)
        
        for i, metric in enumerate(metric_names):
            scores = [nft_scores[i], ft_scores[i], improved_scores[i]]
            best_idx = np.argmax(scores)
            best_model = ['NFT', 'FT', 'Improved'][best_idx]
            print(f"{metric:<20} {nft_scores[i]:<12.2f} "
                  f"{ft_scores[i]:<12.2f} {improved_scores[i]:<12.2f} "
                  f"{best_model:<8}")
        
        # Overall averages
        nft_avg = np.mean(nft_scores)
        ft_avg = np.mean(ft_scores)
        improved_avg = np.mean(improved_scores)
        
        print("-"*70)
        print(f"{'OVERALL AVERAGE':<20} {nft_avg:<12.2f} "
              f"{ft_avg:<12.2f} {improved_avg:<12.2f}")
        
        best_overall = np.argmax([nft_avg, ft_avg, improved_avg])
        best_model_name = ['NFT', 'FT', 'Improved'][best_overall]
        print(f"\n🏆 Best performing model: {best_model_name}")
        
        # Show improvements
        print(f"📊 NFT vs FT difference: {nft_avg - ft_avg:+.2f} points")
        print(f"📊 Improved vs NFT: {improved_avg - nft_avg:+.2f} points")
        print(f"📊 Improved vs FT: {improved_avg - ft_avg:+.2f} points")
        
    else:
        print(f"{'Metric':<25} {'NFT':<12} {'FT':<12} {'Difference':<12}")
        print("-"*60)
        
        for i, metric in enumerate(metric_names):
            diff = nft_scores[i] - ft_scores[i]
            print(f"{metric:<25} {nft_scores[i]:<12.2f} "
                  f"{ft_scores[i]:<12.2f} {diff:+12.2f}")
        
        # Overall averages
        nft_avg = np.mean(nft_scores)
        ft_avg = np.mean(ft_scores)
        
        print("-"*60)
        print(f"{'OVERALL AVERAGE':<25} {nft_avg:<12.2f} "
              f"{ft_avg:<12.2f} {nft_avg - ft_avg:+12.2f}")
        
        better_model = 'NFT' if nft_avg > ft_avg else 'FT'
        print(f"\n🏆 Better performing model: {better_model}")
        avg_improvement = abs(nft_avg - ft_avg)
        print(f"📊 Average improvement: {avg_improvement:.2f} "
              f"percentage points")


if __name__ == "__main__":
    import sys
    
    # Check if user wants to include improved results
    include_improved = False
    if (len(sys.argv) > 1 and
            sys.argv[1].lower() in ['improved', 'all', 'three']):
        include_improved = True
        print("Running comparison with improved Mermaid outputs...")
    else:
        print("Running NFT vs FT comparison...")
        print("Use 'python plot_comparison.py improved' to include "
              "improved results")
    
    plot_metric_comparison(include_improved=include_improved)
