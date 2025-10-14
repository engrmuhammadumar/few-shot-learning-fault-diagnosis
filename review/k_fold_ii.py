# Core Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import os
from PIL import Image
import warnings
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FewShotDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.transform = transform
        self.classes = sorted(os.listdir(dataset_path))
        self.data = []
        self.labels = []
        
        for idx, cls in enumerate(self.classes):
            class_path = os.path.join(dataset_path, cls)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        self.data.append(os.path.join(class_path, img_name))
                        self.labels.append(idx)
        
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class PrototypicalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        
    def forward(self, support_x, support_y, query_x, n_way):
        support_emb = self.flatten(self.backbone(support_x))
        query_emb = self.flatten(self.backbone(query_x))
        prototypes = torch.stack([support_emb[support_y == i].mean(0) for i in range(n_way)])
        dists = torch.cdist(query_emb, prototypes)
        return -dists

def create_episode(dataset, indices, n_way, k_shot, n_query):
    """Create support and query sets for one episode"""
    support_x, support_y, query_x, query_y = [], [], [], []
    
    for class_id in range(n_way):
        class_indices = indices[dataset.labels[indices] == class_id]
        np.random.shuffle(class_indices)
        
        for idx in class_indices[:k_shot]:
            img, _ = dataset[idx]
            support_x.append(img)
            support_y.append(class_id)
        
        for idx in class_indices[k_shot:k_shot + n_query]:
            img, _ = dataset[idx]
            query_x.append(img)
            query_y.append(class_id)
    
    return (torch.stack(support_x), torch.tensor(support_y),
            torch.stack(query_x), torch.tensor(query_y))

def train_fold(model, optimizer, dataset, train_idx, n_way, k_shot, n_query, 
               n_episodes, device):
    """Train one fold"""
    model.train()
    losses, accs = [], []
    
    for episode in range(n_episodes):
        sup_x, sup_y, qry_x, qry_y = create_episode(dataset, train_idx, n_way, k_shot, n_query)
        sup_x, sup_y = sup_x.to(device), sup_y.to(device)
        qry_x, qry_y = qry_x.to(device), qry_y.to(device)
        
        logits = model(sup_x, sup_y, qry_x, n_way)
        loss = nn.CrossEntropyLoss()(logits, qry_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == qry_y).float().mean().item()
        
        losses.append(loss.item())
        accs.append(acc)
        
        if (episode + 1) % 100 == 0:
            print(f"    Episode {episode+1}/{n_episodes} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
    
    return losses, accs

def evaluate_fold_detailed(model, dataset, test_idx, n_way, k_shot, n_query, device):
    """Comprehensive evaluation with all metrics"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for _ in range(20):  # More episodes for robust metrics
            sup_x, sup_y, qry_x, qry_y = create_episode(dataset, test_idx, n_way, k_shot, n_query)
            sup_x, sup_y = sup_x.to(device), sup_y.to(device)
            qry_x, qry_y = qry_x.to(device), qry_y.to(device)
            
            logits = model(sup_x, sup_y, qry_x, n_way)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(qry_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    return acc, cm, all_preds, all_labels, np.array(all_probs), precision, recall, f1

def kfold_cross_validation(dataset, n_way, k_shot, n_query, n_episodes, 
                           n_splits, device, save_dir='kfold_results'):
    """Perform k-fold cross-validation with comprehensive saving"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*70}")
    print(f"K-FOLD CROSS-VALIDATION ({n_splits} folds)")
    print(f"Saving to: {save_dir}")
    print(f"{'='*70}\n")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {
        'train_acc': [], 'val_acc': [], 'train_loss': [],
        'confusion_matrices': [], 'train_histories': [],
        'precision': [], 'recall': [], 'f1_score': [],
        'all_preds': [], 'all_labels': [], 'all_probs': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), dataset.labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{n_splits}")
        print(f"{'='*60}")
        print(f"Train: {len(train_idx)} samples | Val: {len(val_idx)} samples")
        
        # Initialize model
        model = PrototypicalNetwork().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Train
        losses, accs = train_fold(model, optimizer, dataset, train_idx, 
                                  n_way, k_shot, n_query, n_episodes, device)
        
        # Evaluate
        val_acc, cm, preds, labels, probs, prec, rec, f1 = evaluate_fold_detailed(
            model, dataset, val_idx, n_way, k_shot, n_query, device
        )
        
        # Store results
        results['train_acc'].append(np.mean(accs[-50:]))
        results['val_acc'].append(val_acc)
        results['train_loss'].append(np.mean(losses[-50:]))
        results['confusion_matrices'].append(cm.tolist())
        results['train_histories'].append({'loss': losses, 'acc': accs})
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['f1_score'].append(f1)
        results['all_preds'].append(preds)
        results['all_labels'].append(labels)
        results['all_probs'].append(probs)
        
        print(f"\n  Final Metrics:")
        print(f"    Train Acc: {results['train_acc'][-1]:.4f}")
        print(f"    Val Acc:   {val_acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1-Score:  {f1:.4f}")
        
        # Save model
        model_path = os.path.join(save_dir, f'model_fold_{fold+1}.pth')
        torch.save({
            'fold': fold + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_acc,
        }, model_path)
        print(f"    ✓ Model saved: {model_path}")
        
        # Save results after each fold
        results_path = os.path.join(save_dir, 'kfold_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"    ✓ Progress saved: {results_path}")
    
    return results, save_dir

def plot_all_results(results, class_names, n_splits, save_dir):
    """Generate ALL possible plots for publication"""
    
    print(f"\n{'='*70}")
    print("GENERATING ALL PLOTS")
    print(f"{'='*70}\n")
    
    x = np.arange(1, n_splits + 1)
    width = 0.35
    
    # ========== PLOT 1: Accuracy Comparison ==========
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, results['train_acc'], width, label='Training', alpha=0.8, color='steelblue')
    plt.bar(x + width/2, results['val_acc'], width, label='Validation', alpha=0.8, color='coral')
    plt.xlabel('Fold', fontweight='bold', fontsize=14)
    plt.ylabel('Accuracy', fontweight='bold', fontsize=14)
    plt.title('Training vs Validation Accuracy Across Folds', fontweight='bold', fontsize=16)
    plt.legend(fontsize=12)
    plt.xticks(x)
    plt.ylim([0.9, 1.02])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot1_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 1: Accuracy Comparison")
    
    # ========== PLOT 2: Training Loss ==========
    plt.figure(figsize=(10, 6))
    plt.bar(x, results['train_loss'], alpha=0.8, color='crimson', edgecolor='darkred', linewidth=1.5)
    plt.xlabel('Fold', fontweight='bold', fontsize=14)
    plt.ylabel('Loss', fontweight='bold', fontsize=14)
    plt.title('Training Loss Across Folds', fontweight='bold', fontsize=16)
    plt.xticks(x)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot2_training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 2: Training Loss")
    
    # ========== PLOT 3: Training Curves (All Folds) ==========
    plt.figure(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, hist in enumerate(results['train_histories']):
        plt.plot(hist['acc'], alpha=0.7, linewidth=2, label=f'Fold {i+1}', color=colors[i])
    plt.xlabel('Episode', fontweight='bold', fontsize=14)
    plt.ylabel('Training Accuracy', fontweight='bold', fontsize=14)
    plt.title('Training Progress Across All Folds', fontweight='bold', fontsize=16)
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot3_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 3: Training Curves")
    
    # ========== PLOT 4: Loss Curves (All Folds) ==========
    plt.figure(figsize=(12, 7))
    for i, hist in enumerate(results['train_histories']):
        plt.plot(hist['loss'], alpha=0.7, linewidth=2, label=f'Fold {i+1}', color=colors[i])
    plt.xlabel('Episode', fontweight='bold', fontsize=14)
    plt.ylabel('Training Loss', fontweight='bold', fontsize=14)
    plt.title('Loss Progression Across All Folds', fontweight='bold', fontsize=16)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot4_loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 4: Loss Curves")
    
    # ========== PLOT 5: Box Plot Distribution ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Validation Accuracy
    bp1 = axes[0].boxplot([results['val_acc']], tick_labels=['Validation Accuracy'], 
                           patch_artist=True, widths=0.5)
    bp1['boxes'][0].set_facecolor('lightblue')
    bp1['boxes'][0].set_linewidth(2)
    for whisker in bp1['whiskers']:
        whisker.set_linewidth(2)
    axes[0].scatter([1]*len(results['val_acc']), results['val_acc'], 
                    alpha=0.7, s=150, c='red', zorder=3, edgecolors='black', linewidths=1.5)
    axes[0].set_ylabel('Accuracy', fontweight='bold', fontsize=13)
    axes[0].set_title('Validation Accuracy Distribution', fontweight='bold', fontsize=14)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0.95, 1.01])
    
    # Training Loss
    bp2 = axes[1].boxplot([results['train_loss']], tick_labels=['Training Loss'], 
                           patch_artist=True, widths=0.5)
    bp2['boxes'][0].set_facecolor('lightcoral')
    bp2['boxes'][0].set_linewidth(2)
    for whisker in bp2['whiskers']:
        whisker.set_linewidth(2)
    axes[1].scatter([1]*len(results['train_loss']), results['train_loss'], 
                    alpha=0.7, s=150, c='darkred', zorder=3, edgecolors='black', linewidths=1.5)
    axes[1].set_ylabel('Loss', fontweight='bold', fontsize=13)
    axes[1].set_title('Training Loss Distribution', fontweight='bold', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot5_boxplot_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 5: Box Plot Distribution")
    
    # ========== PLOT 6: Mean Confusion Matrix ==========
    plt.figure(figsize=(10, 8))
    mean_cm = np.mean(results['confusion_matrices'], axis=0)
    sns.heatmap(mean_cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Average Count'},
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    plt.xlabel('Predicted Label', fontweight='bold', fontsize=14)
    plt.ylabel('True Label', fontweight='bold', fontsize=14)
    plt.title('Mean Confusion Matrix (K-Fold Cross-Validation)', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot6_mean_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 6: Mean Confusion Matrix")
    
    # ========== PLOT 7: Std Dev Confusion Matrix ==========
    plt.figure(figsize=(10, 8))
    std_cm = np.std(results['confusion_matrices'], axis=0)
    sns.heatmap(std_cm, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Standard Deviation'},
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    plt.xlabel('Predicted Label', fontweight='bold', fontsize=14)
    plt.ylabel('True Label', fontweight='bold', fontsize=14)
    plt.title('Std Dev Confusion Matrix (K-Fold Cross-Validation)', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot7_std_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 7: Std Dev Confusion Matrix")
    
    # ========== PLOT 8: Individual Fold Confusion Matrices ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, cm in enumerate(results['confusion_matrices']):
        sns.heatmap(np.array(cm), annot=True, fmt='.0f', cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[i], cbar_kws={'label': 'Count'},
                    annot_kws={'fontsize': 10, 'fontweight': 'bold'})
        axes[i].set_xlabel('Predicted', fontweight='bold', fontsize=11)
        axes[i].set_ylabel('True', fontweight='bold', fontsize=11)
        axes[i].set_title(f'Fold {i+1} - Acc: {results["val_acc"][i]:.4f}', 
                         fontweight='bold', fontsize=12)
    
    # Hide the 6th subplot
    axes[5].axis('off')
    
    plt.suptitle('Individual Confusion Matrices for Each Fold', fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot8_individual_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 8: Individual Confusion Matrices")
    
    # ========== PLOT 9: Precision, Recall, F1-Score Comparison ==========
    fig, ax = plt.subplots(figsize=(12, 7))
    x_pos = np.arange(n_splits)
    width = 0.25
    
    ax.bar(x_pos - width, results['precision'], width, label='Precision', alpha=0.8, color='skyblue')
    ax.bar(x_pos, results['recall'], width, label='Recall', alpha=0.8, color='lightcoral')
    ax.bar(x_pos + width, results['f1_score'], width, label='F1-Score', alpha=0.8, color='lightgreen')
    
    ax.set_xlabel('Fold', fontweight='bold', fontsize=14)
    ax.set_ylabel('Score', fontweight='bold', fontsize=14)
    ax.set_title('Precision, Recall, and F1-Score Across Folds', fontweight='bold', fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(n_splits)])
    ax.legend(fontsize=12)
    ax.set_ylim([0.9, 1.02])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot9_precision_recall_f1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 9: Precision, Recall, F1-Score")
    
    # ========== PLOT 10: ROC Curves ==========
    plt.figure(figsize=(12, 9))
    n_classes = len(class_names)
    
    # Aggregate all predictions from all folds
    all_labels_combined = np.concatenate(results['all_labels'])
    all_probs_combined = np.vstack(results['all_probs'])
    
    # Binarize labels
    y_true_bin = label_binarize(all_labels_combined, classes=list(range(n_classes)))
    
    # Plot ROC for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs_combined[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=13)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=13)
    plt.title('ROC Curves for All Classes (K-Fold Aggregated)', fontweight='bold', fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot10_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 10: ROC Curves")
    
    # ========== PLOT 11: Combined Summary ==========
    fig = plt.figure(figsize=(20, 12))
    
    # Accuracy comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.bar(x - width/2, results['train_acc'], width, label='Train', alpha=0.8)
    ax1.bar(x + width/2, results['val_acc'], width, label='Val', alpha=0.8)
    ax1.set_xlabel('Fold', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Accuracy Across Folds', fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Training curves
    ax2 = plt.subplot(2, 3, 2)
    for i, hist in enumerate(results['train_histories']):
        ax2.plot(hist['acc'], alpha=0.6, label=f'Fold {i+1}')
    ax2.set_xlabel('Episode', fontweight='bold')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('Training Progress', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    
    # Box plot
    ax3 = plt.subplot(2, 3, 3)
    bp = ax3.boxplot([results['val_acc']], tick_labels=['Val'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax3.scatter([1]*len(results['val_acc']), results['val_acc'], 
                alpha=0.6, s=100, c='red', zorder=3)
    ax3.set_ylabel('Accuracy', fontweight='bold')
    ax3.set_title('Accuracy Distribution', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Mean confusion matrix
    ax4 = plt.subplot(2, 3, 4)
    mean_cm = np.mean(results['confusion_matrices'], axis=0)
    sns.heatmap(mean_cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax4, cbar=False)
    ax4.set_xlabel('Predicted', fontweight='bold', fontsize=10)
    ax4.set_ylabel('True', fontweight='bold', fontsize=10)
    ax4.set_title('Mean Confusion Matrix', fontweight='bold')
    
    # Metrics comparison
    ax5 = plt.subplot(2, 3, 5)
    metrics_data = [results['precision'], results['recall'], results['f1_score']]
    bp2 = ax5.boxplot(metrics_data, tick_labels=['Precision', 'Recall', 'F1'], patch_artist=True)
    for patch, color in zip(bp2['boxes'], ['lightblue', 'lightcoral', 'lightgreen']):
        patch.set_facecolor(color)
    ax5.set_ylabel('Score', fontweight='bold')
    ax5.set_title('Metrics Distribution', fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # Loss curves
    ax6 = plt.subplot(2, 3, 6)
    for i, hist in enumerate(results['train_histories']):
        ax6.plot(hist['loss'], alpha=0.6, label=f'Fold {i+1}')
    ax6.set_xlabel('Episode', fontweight='bold')
    ax6.set_ylabel('Loss', fontweight='bold')
    ax6.set_title('Loss Progress', fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)
    
    plt.suptitle('K-Fold Cross-Validation Summary', fontweight='bold', fontsize=18, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot11_combined_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Plot 11: Combined Summary")
    
    print(f"\n{'='*70}")
    print(f"✓ All 11 plots saved to: {save_dir}")
    print(f"{'='*70}")

def print_comprehensive_summary(results, n_splits, save_dir):
    """Print and save comprehensive statistical summary"""
    
    summary_text = []
    summary_text.append("\n" + "="*70)
    summary_text.append("COMPREHENSIVE K-FOLD CROSS-VALIDATION SUMMARY")
    summary_text.append("="*70 + "\n")
    
    # Accuracy statistics
    train_acc = np.array(results['train_acc'])
    val_acc = np.array(results['val_acc'])
    
    summary_text.append("ACCURACY STATISTICS:")
    summary_text.append(f"  Training Accuracy:   {train_acc.mean():.4f} ± {train_acc.std():.4f}")
    summary_text.append(f"    Min: {train_acc.min():.4f}, Max: {train_acc.max():.4f}")
    summary_text.append(f"  Validation Accuracy: {val_acc.mean():.4f} ± {val_acc.std():.4f}")
    summary_text.append(f"    Min: {val_acc.min():.4f}, Max: {val_acc.max():.4f}")
    summary_text.append(f"    95% CI: [{val_acc.mean() - 1.96*val_acc.std()/np.sqrt(n_splits):.4f}, "
          f"{val_acc.mean() + 1.96*val_acc.std()/np.sqrt(n_splits):.4f}]")
    
    # Other metrics
    precision = np.array(results['precision'])
    recall = np.array(results['recall'])
    f1 = np.array(results['f1_score'])
    
    summary_text.append(f"\nOTHER METRICS:")
    summary_text.append(f"  Precision: {precision.mean():.4f} ± {precision.std():.4f}")
    summary_text.append(f"  Recall:    {recall.mean():.4f} ± {recall.std():.4f}")
    summary_text.append(f"  F1-Score:  {f1.mean():.4f} ± {f1.std():.4f}")
    
    # Loss statistics
    train_loss = np.array(results['train_loss'])
    summary_text.append(f"\nTRAINING LOSS:")
    summary_text.append(f"  Mean ± Std: {train_loss.mean():.4f} ± {train_loss.std():.4f}")
    summary_text.append(f"  Min: {train_loss.min():.4f}, Max: {train_loss.max():.4f}")
    
    # Fold-by-fold details
    summary_text.append(f"\n{'='*70}")
    summary_text.append("FOLD-BY-FOLD DETAILS:")
    summary_text.append(f"{'='*70}")
    summary_text.append(f"{'Fold':<6} {'Train Acc':<11} {'Val Acc':<11} {'Loss':<10} {'Precision':<11} {'Recall':<11} {'F1':<10}")
    summary_text.append("-" * 70)
    
    for i in range(n_splits):
        summary_text.append(
            f"{i+1:<6} {results['train_acc'][i]:<11.4f} {results['val_acc'][i]:<11.4f} "
            f"{results['train_loss'][i]:<10.4f} {results['precision'][i]:<11.4f} "
            f"{results['recall'][i]:<11.4f} {results['f1_score'][i]:<10.4f}"
        )
    
    summary_text.append("="*70)
    
    # Print to console
    for line in summary_text:
        print(line)
    
    # Save to file
    summary_file = os.path.join(save_dir, 'summary_statistics.txt')
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_text))
    
    print(f"\n✓ Summary saved to: {summary_file}\n")

# MAIN EXECUTION
if __name__ == "__main__":
    print("\n" + "="*70)
    print("K-FOLD CROSS-VALIDATION WITH COMPREHENSIVE ANALYSIS")
    print("="*70)
    
    # Configuration
    DATASET_PATH = r"E:\1 Paper MCT\Cutting Tool Paper\Dataset\cutting tool data\test_data_40_images"
    N_WAY = 7
    K_SHOT = 5
    N_QUERY = 15
    N_EPISODES = 500
    N_SPLITS = 5
    CLASS_NAMES = ['BF', 'BFI', 'GF', 'GFI', 'N', 'NI', 'TF']
    SAVE_DIR = 'kfold_results'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  N-way K-shot: {N_WAY}-way {K_SHOT}-shot")
    print(f"  Query size: {N_QUERY}")
    print(f"  Episodes per fold: {N_EPISODES}")
    print(f"  Number of folds: {N_SPLITS}")
    print(f"  Save directory: {SAVE_DIR}")
    
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = FewShotDataset(DATASET_PATH, transform=transform)
    print(f"✓ Dataset loaded: {len(dataset)} samples, {len(dataset.classes)} classes")
    print(f"  Classes: {CLASS_NAMES}")
    
    # Run k-fold cross-validation
    results, save_dir = kfold_cross_validation(
        dataset, N_WAY, K_SHOT, N_QUERY, N_EPISODES, N_SPLITS, device, SAVE_DIR
    )
    
    # Print comprehensive summary
    print_comprehensive_summary(results, N_SPLITS, save_dir)
    
    # Generate all plots
    plot_all_results(results, CLASS_NAMES, N_SPLITS, save_dir)
    
    # Final message
    print("\n" + "="*70)
    print("✓✓✓ K-FOLD CROSS-VALIDATION COMPLETED SUCCESSFULLY ✓✓✓")
    print("="*70)
    print(f"\nAll results saved to: {save_dir}/")
    print(f"  • 5 trained models (model_fold_1.pth to model_fold_5.pth)")
    print(f"  • Complete results (kfold_results.pkl)")
    print(f"  • Summary statistics (summary_statistics.txt)")
    print(f"  • 11 publication-quality plots")
    print("\nFor paper:")
    print(f"  Mean Validation Accuracy: {np.mean(results['val_acc']):.2%} ± {np.std(results['val_acc']):.2%}")
    print(f"  Mean Precision: {np.mean(results['precision']):.2%}")
    print(f"  Mean Recall: {np.mean(results['recall']):.2%}")
    print(f"  Mean F1-Score: {np.mean(results['f1_score']):.2%}")
    print("="*70 + "\n")