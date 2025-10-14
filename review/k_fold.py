# Core Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import os
from PIL import Image
import warnings
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
        # Get embeddings
        support_emb = self.flatten(self.backbone(support_x))
        query_emb = self.flatten(self.backbone(query_x))
        
        # Compute prototypes
        prototypes = torch.stack([support_emb[support_y == i].mean(0) for i in range(n_way)])
        
        # Euclidean distance
        dists = torch.cdist(query_emb, prototypes)
        return -dists  # Negative distance for classification

def create_episode(dataset, indices, n_way, k_shot, n_query):
    """Create support and query sets for one episode"""
    support_x, support_y, query_x, query_y = [], [], [], []
    
    for class_id in range(n_way):
        class_indices = indices[dataset.labels[indices] == class_id]
        np.random.shuffle(class_indices)
        
        # Support set
        for idx in class_indices[:k_shot]:
            img, _ = dataset[idx]
            support_x.append(img)
            support_y.append(class_id)
        
        # Query set
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
        # Create episode
        sup_x, sup_y, qry_x, qry_y = create_episode(dataset, train_idx, n_way, k_shot, n_query)
        sup_x, sup_y = sup_x.to(device), sup_y.to(device)
        qry_x, qry_y = qry_x.to(device), qry_y.to(device)
        
        # Forward
        logits = model(sup_x, sup_y, qry_x, n_way)
        loss = nn.CrossEntropyLoss()(logits, qry_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == qry_y).float().mean().item()
        
        losses.append(loss.item())
        accs.append(acc)
        
        if (episode + 1) % 100 == 0:
            print(f"    Episode {episode+1}/{n_episodes} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
    
    return losses, accs

def evaluate_fold(model, dataset, test_idx, n_way, k_shot, n_query, device):
    """Evaluate one fold"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        # Multiple episodes for robust evaluation
        for _ in range(10):
            sup_x, sup_y, qry_x, qry_y = create_episode(dataset, test_idx, n_way, k_shot, n_query)
            sup_x, sup_y = sup_x.to(device), sup_y.to(device)
            qry_x, qry_y = qry_x.to(device), qry_y.to(device)
            
            logits = model(sup_x, sup_y, qry_x, n_way)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(qry_y.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    return acc, cm, all_preds, all_labels

def kfold_cross_validation(dataset, n_way, k_shot, n_query, n_episodes, 
                           n_splits, device):
    """Perform k-fold cross-validation"""
    print(f"\n{'='*70}")
    print(f"K-FOLD CROSS-VALIDATION ({n_splits} folds)")
    print(f"{'='*70}\n")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {
        'train_acc': [], 'val_acc': [], 'train_loss': [],
        'confusion_matrices': [], 'train_histories': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), dataset.labels)):
        print(f"\nFOLD {fold + 1}/{n_splits}")
        print(f"Train: {len(train_idx)} samples | Val: {len(val_idx)} samples")
        
        # Initialize model
        model = PrototypicalNetwork().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Train
        losses, accs = train_fold(model, optimizer, dataset, train_idx, 
                                  n_way, k_shot, n_query, n_episodes, device)
        
        # Evaluate
        val_acc, cm, _, _ = evaluate_fold(model, dataset, val_idx, n_way, 
                                          k_shot, n_query, device)
        
        # Store results
        results['train_acc'].append(np.mean(accs[-50:]))
        results['val_acc'].append(val_acc)
        results['train_loss'].append(np.mean(losses[-50:]))
        results['confusion_matrices'].append(cm)
        results['train_histories'].append({'loss': losses, 'acc': accs})
        
        print(f"  Train Acc: {results['train_acc'][-1]:.4f} | Val Acc: {val_acc:.4f}")
    
    return results

def plot_results(results, class_names, n_splits):
    """Plot comprehensive results"""
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Accuracy across folds
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(1, n_splits + 1)
    width = 0.35
    ax1.bar(x - width/2, results['train_acc'], width, label='Train', alpha=0.8)
    ax1.bar(x + width/2, results['val_acc'], width, label='Validation', alpha=0.8)
    ax1.set_xlabel('Fold', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Accuracy Across Folds', fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Loss across folds
    ax2 = plt.subplot(2, 3, 2)
    ax2.bar(x, results['train_loss'], alpha=0.8, color='coral')
    ax2.set_xlabel('Fold', fontweight='bold')
    ax2.set_ylabel('Loss', fontweight='bold')
    ax2.set_title('Training Loss Across Folds', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Training curves
    ax3 = plt.subplot(2, 3, 3)
    for i, hist in enumerate(results['train_histories']):
        ax3.plot(hist['acc'], alpha=0.6, label=f'Fold {i+1}')
    ax3.set_xlabel('Episode', fontweight='bold')
    ax3.set_ylabel('Accuracy', fontweight='bold')
    ax3.set_title('Training Progress', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Box plot
    ax4 = plt.subplot(2, 3, 4)
    bp = ax4.boxplot([results['val_acc']], labels=['Validation'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax4.scatter([1]*len(results['val_acc']), results['val_acc'], 
                alpha=0.6, s=100, c='red', zorder=3)
    ax4.set_ylabel('Accuracy', fontweight='bold')
    ax4.set_title('Validation Accuracy Distribution', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Mean confusion matrix
    ax5 = plt.subplot(2, 3, 5)
    mean_cm = np.mean(results['confusion_matrices'], axis=0)
    sns.heatmap(mean_cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax5)
    ax5.set_xlabel('Predicted', fontweight='bold')
    ax5.set_ylabel('True', fontweight='bold')
    ax5.set_title('Mean Confusion Matrix', fontweight='bold')
    
    # 6. Std confusion matrix
    ax6 = plt.subplot(2, 3, 6)
    std_cm = np.std(results['confusion_matrices'], axis=0)
    sns.heatmap(std_cm, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names, ax=ax6)
    ax6.set_xlabel('Predicted', fontweight='bold')
    ax6.set_ylabel('True', fontweight='bold')
    ax6.set_title('Std Dev Confusion Matrix', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('kfold_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(results, n_splits):
    """Print statistical summary"""
    print(f"\n{'='*70}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*70}")
    
    train_acc = np.array(results['train_acc'])
    val_acc = np.array(results['val_acc'])
    
    print(f"\nTraining Accuracy:   {train_acc.mean():.4f} ± {train_acc.std():.4f}")
    print(f"Validation Accuracy: {val_acc.mean():.4f} ± {val_acc.std():.4f}")
    print(f"95% CI: [{val_acc.mean() - 1.96*val_acc.std()/np.sqrt(n_splits):.4f}, "
          f"{val_acc.mean() + 1.96*val_acc.std()/np.sqrt(n_splits):.4f}]")
    
    print(f"\n{'Fold':<8} {'Train Acc':<12} {'Val Acc':<12} {'Loss':<10}")
    print("-" * 45)
    for i in range(n_splits):
        print(f"{i+1:<8} {results['train_acc'][i]:<12.4f} "
              f"{results['val_acc'][i]:<12.4f} {results['train_loss'][i]:<10.4f}")
    print(f"{'='*70}\n")

# MAIN EXECUTION
if __name__ == "__main__":
    # Configuration
    DATASET_PATH = r"E:\1 Paper MCT\Cutting Tool Paper\Dataset\cutting tool data\test_data_40_images"
    N_WAY = 7
    K_SHOT = 5
    N_QUERY = 15
    N_EPISODES = 500
    N_SPLITS = 5
    CLASS_NAMES = ['BF', 'BFI', 'GF', 'GFI', 'N', 'NI', 'TF']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Configuration: {N_WAY}-way {K_SHOT}-shot | Query: {N_QUERY}")
    print(f"Episodes per fold: {N_EPISODES} | K-folds: {N_SPLITS}")
    
    # Load dataset
    dataset = FewShotDataset(DATASET_PATH, transform=transform)
    print(f"\nDataset: {len(dataset)} samples, {len(dataset.classes)} classes")
    
    # Run k-fold cross-validation
    results = kfold_cross_validation(dataset, N_WAY, K_SHOT, N_QUERY, 
                                     N_EPISODES, N_SPLITS, device)
    
    # Results
    print_summary(results, N_SPLITS)
    plot_results(results, CLASS_NAMES, N_SPLITS)
    
    print("✓ K-fold cross-validation completed!")
    print("✓ Results saved as 'kfold_results.png'")