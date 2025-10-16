"""
Regenerate all plots with bold styling and no titles
Run this to create publication-ready figures
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import pickle
import os

# Set global matplotlib parameters for bold fonts
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['axes.labelsize'] = 13

def load_results(results_path='kfold_results/kfold_results.pkl'):
    """Load saved results"""
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results

def plot_all_bold_notitle(results, class_names, save_dir='kfold_results_bold'):
    """Generate all plots with bold styling and no titles"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("REGENERATING ALL PLOTS - BOLD STYLE, NO TITLES")
    print(f"{'='*70}\n")
    
    n_splits = len(results['train_acc'])
    x = np.arange(1, n_splits + 1)
    width = 0.35
    
    # ========== PLOT 1: Accuracy Comparison ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, results['train_acc'], width, label='Training', 
                   alpha=0.85, color='steelblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, results['val_acc'], width, label='Validation', 
                   alpha=0.85, color='coral', edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Fold', fontweight='bold', fontsize=14)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=14)
    legend = ax.legend(fontsize=13, frameon=True, edgecolor='black')
    plt.setp(legend.get_texts(), fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontweight='bold', fontsize=12)
    ax.set_yticklabels([f'{val:.2f}' for val in ax.get_yticks()], fontweight='bold', fontsize=12)
    ax.set_ylim([0.9, 1.02])
    ax.grid(axis='y', alpha=0.3, linewidth=1.2)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot1_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 1: Accuracy Comparison")
    
    # ========== PLOT 2: Training Loss ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, results['train_loss'], alpha=0.85, color='crimson', 
                  edgecolor='darkred', linewidth=2)
    ax.set_xlabel('Fold', fontweight='bold', fontsize=14)
    ax.set_ylabel('Loss', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontweight='bold', fontsize=12)
    ax.set_yticklabels([f'{val:.4f}' for val in ax.get_yticks()], fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linewidth=1.2)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot2_training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 2: Training Loss")
    
    # ========== PLOT 3: Training Curves ==========
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, hist in enumerate(results['train_histories']):
        ax.plot(hist['acc'], alpha=0.8, linewidth=2.5, label=f'Fold {i+1}', color=colors[i])
    ax.set_xlabel('Episode', fontweight='bold', fontsize=14)
    ax.set_ylabel('Training Accuracy', fontweight='bold', fontsize=14)
    legend = ax.legend(fontsize=12, frameon=True, edgecolor='black', loc='lower right')
    plt.setp(legend.get_texts(), fontweight='bold')
    ax.set_xticklabels([f'{int(val)}' for val in ax.get_xticks()], fontweight='bold', fontsize=12)
    ax.set_yticklabels([f'{val:.2f}' for val in ax.get_yticks()], fontweight='bold', fontsize=12)
    ax.grid(alpha=0.3, linewidth=1.2)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot3_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 3: Training Curves")
    
    # ========== PLOT 4: Loss Curves ==========
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, hist in enumerate(results['train_histories']):
        ax.plot(hist['loss'], alpha=0.8, linewidth=2.5, label=f'Fold {i+1}', color=colors[i])
    ax.set_xlabel('Episode', fontweight='bold', fontsize=14)
    ax.set_ylabel('Training Loss', fontweight='bold', fontsize=14)
    legend = ax.legend(fontsize=12, frameon=True, edgecolor='black', loc='upper right')
    plt.setp(legend.get_texts(), fontweight='bold')
    ax.set_xticklabels([f'{int(val)}' for val in ax.get_xticks()], fontweight='bold', fontsize=12)
    ax.set_yticklabels([f'{val:.2f}' for val in ax.get_yticks()], fontweight='bold', fontsize=12)
    ax.grid(alpha=0.3, linewidth=1.2)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot4_loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 4: Loss Curves")
    
    # ========== PLOT 5: Box Plot Distribution ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Validation Accuracy
    bp1 = axes[0].boxplot([results['val_acc']], tick_labels=['Validation Accuracy'], 
                           patch_artist=True, widths=0.5,
                           boxprops=dict(linewidth=2, edgecolor='black'),
                           whiskerprops=dict(linewidth=2),
                           capprops=dict(linewidth=2),
                           medianprops=dict(linewidth=2.5, color='darkblue'))
    bp1['boxes'][0].set_facecolor('lightblue')
    axes[0].scatter([1]*len(results['val_acc']), results['val_acc'], 
                    alpha=0.8, s=180, c='red', zorder=3, edgecolors='black', linewidths=2)
    axes[0].set_ylabel('Accuracy', fontweight='bold', fontsize=14)
    axes[0].set_xticklabels(['Validation Accuracy'], fontweight='bold', fontsize=12)
    axes[0].set_yticklabels([f'{val:.2f}' for val in axes[0].get_yticks()], fontweight='bold', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3, linewidth=1.2)
    axes[0].set_ylim([0.95, 1.01])
    for spine in axes[0].spines.values():
        spine.set_linewidth(1.5)
    
    # Training Loss
    bp2 = axes[1].boxplot([results['train_loss']], tick_labels=['Training Loss'], 
                           patch_artist=True, widths=0.5,
                           boxprops=dict(linewidth=2, edgecolor='black'),
                           whiskerprops=dict(linewidth=2),
                           capprops=dict(linewidth=2),
                           medianprops=dict(linewidth=2.5, color='darkred'))
    bp2['boxes'][0].set_facecolor('lightcoral')
    axes[1].scatter([1]*len(results['train_loss']), results['train_loss'], 
                    alpha=0.8, s=180, c='darkred', zorder=3, edgecolors='black', linewidths=2)
    axes[1].set_ylabel('Loss', fontweight='bold', fontsize=14)
    axes[1].set_xticklabels(['Training Loss'], fontweight='bold', fontsize=12)
    axes[1].set_yticklabels([f'{val:.4f}' for val in axes[1].get_yticks()], fontweight='bold', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3, linewidth=1.2)
    for spine in axes[1].spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot5_boxplot_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 5: Box Plot Distribution")
    
    # ========== PLOT 6: Mean Confusion Matrix ==========
    fig, ax = plt.subplots(figsize=(10, 8))
    mean_cm = np.mean(results['confusion_matrices'], axis=0)
    sns.heatmap(mean_cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Average Count'},
                annot_kws={'fontsize': 13, 'fontweight': 'bold'},
                linewidths=2, linecolor='black', ax=ax)
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=14)
    ax.set_ylabel('True Label', fontweight='bold', fontsize=14)
    ax.set_xticklabels(class_names, fontweight='bold', fontsize=12, rotation=0)
    ax.set_yticklabels(class_names, fontweight='bold', fontsize=12, rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Average Count', fontweight='bold', fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot6_mean_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 6: Mean Confusion Matrix")
    
    # ========== PLOT 7: Std Dev Confusion Matrix ==========
    fig, ax = plt.subplots(figsize=(10, 8))
    std_cm = np.std(results['confusion_matrices'], axis=0)
    sns.heatmap(std_cm, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Standard Deviation'},
                annot_kws={'fontsize': 13, 'fontweight': 'bold'},
                linewidths=2, linecolor='black', ax=ax)
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=14)
    ax.set_ylabel('True Label', fontweight='bold', fontsize=14)
    ax.set_xticklabels(class_names, fontweight='bold', fontsize=12, rotation=0)
    ax.set_yticklabels(class_names, fontweight='bold', fontsize=12, rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Standard Deviation', fontweight='bold', fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot7_std_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 7: Std Dev Confusion Matrix")
    
    # ========== PLOT 8: Individual Confusion Matrices ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, cm in enumerate(results['confusion_matrices']):
        sns.heatmap(np.array(cm), annot=True, fmt='.0f', cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[i], cbar_kws={'label': 'Count'},
                    annot_kws={'fontsize': 11, 'fontweight': 'bold'},
                    linewidths=1.5, linecolor='black')
        axes[i].set_xlabel('Predicted', fontweight='bold', fontsize=12)
        axes[i].set_ylabel('True', fontweight='bold', fontsize=12)
        axes[i].set_xticklabels(class_names, fontweight='bold', fontsize=10, rotation=0)
        axes[i].set_yticklabels(class_names, fontweight='bold', fontsize=10, rotation=0)
        
        # Make fold label bold
        fold_text = f'Fold {i+1}: {results["val_acc"][i]:.4f}'
        axes[i].text(0.5, 1.05, fold_text, transform=axes[i].transAxes,
                    fontsize=13, fontweight='bold', ha='center')
    
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot8_individual_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 8: Individual Confusion Matrices")
    
    # ========== PLOT 9: Precision, Recall, F1-Score ==========
    fig, ax = plt.subplots(figsize=(12, 7))
    x_pos = np.arange(n_splits)
    width = 0.25
    
    bars1 = ax.bar(x_pos - width, results['precision'], width, label='Precision', 
                   alpha=0.85, color='skyblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos, results['recall'], width, label='Recall', 
                   alpha=0.85, color='lightcoral', edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x_pos + width, results['f1_score'], width, label='F1-Score', 
                   alpha=0.85, color='lightgreen', edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Fold', fontweight='bold', fontsize=14)
    ax.set_ylabel('Score', fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(n_splits)], fontweight='bold', fontsize=12)
    ax.set_yticklabels([f'{val:.2f}' for val in ax.get_yticks()], fontweight='bold', fontsize=12)
    legend = ax.legend(fontsize=13, frameon=True, edgecolor='black')
    plt.setp(legend.get_texts(), fontweight='bold')
    ax.set_ylim([0.9, 1.02])
    ax.grid(axis='y', alpha=0.3, linewidth=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot9_precision_recall_f1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 9: Precision, Recall, F1-Score")
    
    # ========== PLOT 10: ROC Curves ==========
    fig, ax = plt.subplots(figsize=(12, 9))
    n_classes = len(class_names)
    
    all_labels_combined = np.concatenate(results['all_labels'])
    all_probs_combined = np.vstack(results['all_probs'])
    y_true_bin = label_binarize(all_labels_combined, classes=list(range(n_classes)))
    
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs_combined[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.5, linestyle=line_styles[i],
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2.5, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=14)
    ax.set_xticklabels([f'{val:.1f}' for val in ax.get_xticks()], fontweight='bold', fontsize=12)
    ax.set_yticklabels([f'{val:.1f}' for val in ax.get_yticks()], fontweight='bold', fontsize=12)
    legend = ax.legend(loc="lower right", fontsize=11, frameon=True, edgecolor='black')
    plt.setp(legend.get_texts(), fontweight='bold')
    ax.grid(alpha=0.3, linewidth=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot10_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 10: ROC Curves")
    
    # ========== PLOT 11: Combined Summary (6 subplots) ==========
    fig = plt.figure(figsize=(20, 12))
    
    # Accuracy comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.bar(x - width/2, results['train_acc'], width, label='Train', alpha=0.85, edgecolor='black', linewidth=1.2)
    ax1.bar(x + width/2, results['val_acc'], width, label='Val', alpha=0.85, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Fold', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    legend1 = ax1.legend(fontsize=10, frameon=True, edgecolor='black')
    plt.setp(legend1.get_texts(), fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontweight('bold')
    ax1.grid(axis='y', alpha=0.3, linewidth=1)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.2)
    
    # Training curves
    ax2 = plt.subplot(2, 3, 2)
    for i, hist in enumerate(results['train_histories']):
        ax2.plot(hist['acc'], alpha=0.7, linewidth=2, label=f'Fold {i+1}')
    ax2.set_xlabel('Episode', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    legend2 = ax2.legend(fontsize=9, frameon=True, edgecolor='black')
    plt.setp(legend2.get_texts(), fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontweight('bold')
    ax2.grid(alpha=0.3, linewidth=1)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.2)
    
    # Box plot
    ax3 = plt.subplot(2, 3, 3)
    bp = ax3.boxplot([results['val_acc']], tick_labels=['Val'], patch_artist=True,
                      boxprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5), medianprops=dict(linewidth=2))
    bp['boxes'][0].set_facecolor('lightblue')
    ax3.scatter([1]*len(results['val_acc']), results['val_acc'], 
                alpha=0.7, s=100, c='red', zorder=3, edgecolors='black', linewidths=1.5)
    ax3.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_fontweight('bold')
    ax3.grid(axis='y', alpha=0.3, linewidth=1)
    for spine in ax3.spines.values():
        spine.set_linewidth(1.2)
    
    # Mean confusion matrix
    ax4 = plt.subplot(2, 3, 4)
    mean_cm = np.mean(results['confusion_matrices'], axis=0)
    sns.heatmap(mean_cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax4, 
                cbar=False, annot_kws={'fontsize': 9, 'fontweight': 'bold'},
                linewidths=1, linecolor='gray')
    ax4.set_xlabel('Predicted', fontweight='bold', fontsize=11)
    ax4.set_ylabel('True', fontweight='bold', fontsize=11)
    ax4.set_xticklabels(class_names, fontweight='bold', fontsize=9, rotation=0)
    ax4.set_yticklabels(class_names, fontweight='bold', fontsize=9, rotation=0)
    
    # Metrics comparison
    ax5 = plt.subplot(2, 3, 5)
    metrics_data = [results['precision'], results['recall'], results['f1_score']]
    bp2 = ax5.boxplot(metrics_data, tick_labels=['Precision', 'Recall', 'F1'], 
                      patch_artist=True,
                      boxprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5), medianprops=dict(linewidth=2))
    for patch, color in zip(bp2['boxes'], ['lightblue', 'lightcoral', 'lightgreen']):
        patch.set_facecolor(color)
    ax5.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax5.tick_params(axis='both', which='major', labelsize=10)
    for label in ax5.get_xticklabels() + ax5.get_yticklabels():
        label.set_fontweight('bold')
    ax5.grid(axis='y', alpha=0.3, linewidth=1)
    for spine in ax5.spines.values():
        spine.set_linewidth(1.2)
    
    # Loss curves
    ax6 = plt.subplot(2, 3, 6)
    for i, hist in enumerate(results['train_histories']):
        ax6.plot(hist['loss'], alpha=0.7, linewidth=2, label=f'Fold {i+1}')
    ax6.set_xlabel('Episode', fontweight='bold', fontsize=12)
    ax6.set_ylabel('Loss', fontweight='bold', fontsize=12)
    legend6 = ax6.legend(fontsize=9, frameon=True, edgecolor='black')
    plt.setp(legend6.get_texts(), fontweight='bold')
    ax6.tick_params(axis='both', which='major', labelsize=10)
    for label in ax6.get_xticklabels() + ax6.get_yticklabels():
        label.set_fontweight('bold')
    ax6.grid(alpha=0.3, linewidth=1)
    for spine in ax6.spines.values():
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot11_combined_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 11: Combined Summary")
    
    print(f"\n{'='*70}")
    print(f"✓ All 11 bold plots (no titles) saved to: {save_dir}/")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("REGENERATING PLOTS - BOLD STYLE, NO TITLES")
    print("="*70)
    
    # Configuration
    CLASS_NAMES = ['BF', 'BFI', 'GF', 'GFI', 'N', 'NI', 'TF']
    RESULTS_PATH = 'kfold_results/kfold_results.pkl'
    SAVE_DIR = 'kfold_results_bold'
    
    # Load results
    print(f"\nLoading results from: {RESULTS_PATH}")
    try:
        results = load_results(RESULTS_PATH)
        print(f"✓ Results loaded successfully")
    except FileNotFoundError:
        print(f"✗ Error: {RESULTS_PATH} not found!")
        print("  Make sure you've run the k-fold training first.")
        exit()
    
    # Generate all plots
    plot_all_bold_notitle(results, CLASS_NAMES, SAVE_DIR)
    
    print("="*70)
    print("✓✓✓ ALL PLOTS REGENERATED SUCCESSFULLY ✓✓✓")
    print("="*70)
    print(f"\nNew plots saved to: {SAVE_DIR}/")
    print("  • All text is bold")
    print("  • No titles")
    print("  • Publication-ready")
    print("="*70 + "\n")