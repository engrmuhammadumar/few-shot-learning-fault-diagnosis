# Core Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms, models

# Utilities
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from scipy import stats
import json

# ============================================================================
# DATASET CLASSES
# ============================================================================

class FewShotDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.classes = sorted(os.listdir(dataset_path))
        self.image_paths = []
        self.labels = []

        for idx, cls in enumerate(self.classes):
            class_path = os.path.join(dataset_path, cls)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        self.image_paths.append(img_path)
                        self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class ValidationDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        class_folders = {
            'BF': 'BF660_1',
            'GF': 'GF660_1', 
            'N': 'N660_1',
            'TF': 'TF660_1'
        }
        
        self.classes = sorted(class_folders.keys())
        
        for idx, (class_name, folder_name) in enumerate(sorted(class_folders.items())):
            class_path = os.path.join(root_path, folder_name, 'AE')
            
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(idx)
                print(f"Loaded class '{class_name}' from {folder_name}/AE: {len([l for l in self.labels if l == idx])} samples")
            else:
                print(f"WARNING: Path not found: {class_path}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ============================================================================
# MODEL CLASSES
# ============================================================================

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)


class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = backbone

    def compute_prototypes(self, support_embeddings, support_labels, n_way):
        prototypes = []
        for i in range(n_way):
            class_embeddings = support_embeddings[support_labels == i]
            prototypes.append(class_embeddings.mean(0))
        return torch.stack(prototypes)

    def forward(self, support_images, support_labels, query_images, n_way):
        support_embeddings = self.backbone(support_images)
        query_embeddings = self.backbone(query_images)
        prototypes = self.compute_prototypes(support_embeddings, support_labels, n_way)
        dists = torch.cdist(query_embeddings, prototypes)
        return dists

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def split_support_query(dataset, n_way, k_shot, query_size):
    class_to_indices = {cls: [] for cls in range(len(dataset.classes))}
    for idx, (_, label) in enumerate(dataset):
        class_to_indices[label].append(idx)
    
    support_set = []
    query_set = []
    
    for cls in range(n_way):
        indices = class_to_indices[cls]
        if len(indices) < k_shot + query_size:
            print(f"WARNING: Class {cls} has only {len(indices)} samples")
            continue
        support_indices, query_indices = train_test_split(
            indices, train_size=k_shot, test_size=query_size, random_state=42
        )
        support_set.extend(support_indices)
        query_set.extend(query_indices)
    
    return support_set, query_set


def train_prototypical_network(model, optimizer, dataset, support_indices, query_indices, 
                               n_way, num_epochs, device):
    model.to(device)
    loss_history = []
    accuracy_history = []
    
    for epoch in range(num_epochs):
        model.train()
        
        support_images, support_labels = [], []
        query_images, query_labels = [], []
        
        for idx in support_indices:
            img, label = dataset[idx]
            support_images.append(img)
            support_labels.append(label)
        
        for idx in query_indices:
            img, label = dataset[idx]
            query_images.append(img)
            query_labels.append(label)
        
        support_images = torch.stack(support_images).to(device)
        support_labels = torch.tensor(support_labels).to(device)
        query_images = torch.stack(query_images).to(device)
        query_labels = torch.tensor(query_labels).to(device)
        
        dists = model(support_images, support_labels, query_images, n_way)
        log_probs = -dists.log_softmax(dim=1)
        loss = nn.CrossEntropyLoss()(log_probs, query_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            preds = torch.argmin(dists, dim=1)
            accuracy = accuracy_score(query_labels.cpu(), preds.cpu())
        
        loss_history.append(loss.item())
        accuracy_history.append(accuracy)
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
    
    return loss_history, accuracy_history


def evaluate_prototypical_network(model, support_images, support_labels, 
                                 query_images, query_labels, n_way):
    model.eval()
    with torch.no_grad():
        dists = model(support_images, support_labels, query_images, n_way)
        preds = torch.argmin(dists, dim=1).cpu().numpy()
        true_labels = query_labels.cpu().numpy()
        
        accuracy = accuracy_score(true_labels, preds)
        conf_matrix = confusion_matrix(true_labels, preds)
        class_report = classification_report(
            true_labels, preds, 
            target_names=[f"Class {i}" for i in range(n_way)],
            zero_division=0
        )
        
    return accuracy, conf_matrix, class_report


def evaluate_cross_domain(model_7class, dataset_4class, support_indices, query_indices, 
                         n_way_target, device, class_mapping):
    model_7class.eval()
    
    support_images, support_labels_4class = [], []
    query_images, query_labels_4class = [], []
    
    for idx in support_indices:
        img, label = dataset_4class[idx]
        support_images.append(img)
        support_labels_4class.append(label)
    
    for idx in query_indices:
        img, label = dataset_4class[idx]
        query_images.append(img)
        query_labels_4class.append(label)
    
    support_images = torch.stack(support_images).to(device)
    query_images = torch.stack(query_images).to(device)
    
    support_labels_7class = torch.tensor([class_mapping[l] for l in support_labels_4class]).to(device)
    query_labels_4class_tensor = torch.tensor(query_labels_4class)
    
    with torch.no_grad():
        support_embeddings = model_7class.backbone(support_images)
        query_embeddings = model_7class.backbone(query_images)
        
        prototypes = []
        for i in range(n_way_target):
            label_7class = class_mapping[i]
            class_mask = support_labels_7class == label_7class
            if class_mask.sum() > 0:
                class_embeddings = support_embeddings[class_mask]
                prototypes.append(class_embeddings.mean(0))
            else:
                prototypes.append(torch.zeros_like(support_embeddings[0]))
        
        prototypes = torch.stack(prototypes)
        dists = torch.cdist(query_embeddings, prototypes)
        preds_4class = torch.argmin(dists, dim=1).cpu().numpy()
    
    accuracy = accuracy_score(query_labels_4class, preds_4class)
    conf_matrix = confusion_matrix(query_labels_4class, preds_4class)
    
    return accuracy, conf_matrix, preds_4class, query_labels_4class


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # ========================================================================
    # PART 1: Load and train on ORIGINAL 7-class dataset
    # ========================================================================
    
    print("="*80)
    print("PART 1: TRAINING ON ORIGINAL 7-CLASS DATASET")
    print("="*80)
    
    dataset_path_7class = r"E:\1 Paper MCT\Cutting Tool Paper\Dataset\cutting tool data\test_data_40_images"
    dataset_7class = FewShotDataset(dataset_path_7class, transform=transform)
    print(f"7-Class Dataset: {len(dataset_7class)} samples, {len(dataset_7class.classes)} classes")
    print(f"Classes: {dataset_7class.classes}\n")
    
    n_way_7 = 7
    k_shot = 5
    query_size = 15
    
    support_indices_7, query_indices_7 = split_support_query(dataset_7class, n_way_7, k_shot, query_size)
    
    backbone_7 = Backbone()
    model_7class = PrototypicalNetwork(backbone_7).to(device)
    optimizer_7 = optim.Adam(model_7class.parameters(), lr=1e-3)
    
    print("Training 7-class model...")
    loss_hist_7, acc_hist_7 = train_prototypical_network(
        model_7class, optimizer_7, dataset_7class, support_indices_7, query_indices_7,
        n_way_7, 500, device
    )
    
    # ========================================================================
    # PART 2: Load and train on NEW 4-class validation dataset
    # ========================================================================
    
    print("\n" + "="*80)
    print("PART 2: TRAINING ON NEW 4-CLASS VALIDATION DATASET")
    print("="*80)
    
    dataset_path_4class = r"F:\20240925"
    dataset_4class = ValidationDataset(dataset_path_4class, transform=transform)
    print(f"\n4-Class Dataset: {len(dataset_4class)} samples, {len(dataset_4class.classes)} classes")
    print(f"Classes: {dataset_4class.classes}\n")
    
    n_way_4 = 4
    
    support_indices_4, query_indices_4 = split_support_query(dataset_4class, n_way_4, k_shot, query_size)
    
    backbone_4 = Backbone()
    model_4class = PrototypicalNetwork(backbone_4).to(device)
    optimizer_4 = optim.Adam(model_4class.parameters(), lr=1e-3)
    
    print("Training 4-class model...")
    loss_hist_4, acc_hist_4 = train_prototypical_network(
        model_4class, optimizer_4, dataset_4class, support_indices_4, query_indices_4,
        n_way_4, 500, device
    )
    
    # ========================================================================
    # PART 3: Evaluate both models
    # ========================================================================
    
    print("\n" + "="*80)
    print("PART 3: EVALUATION")
    print("="*80)
    
    # Create evaluation splits
    support_eval_4, query_eval_4 = split_support_query(dataset_4class, n_way_4, k_shot, query_size)
    
    # Prepare data
    sup_imgs, sup_lbls = [], []
    qry_imgs, qry_lbls = [], []
    
    for idx in support_eval_4:
        img, label = dataset_4class[idx]
        sup_imgs.append(img)
        sup_lbls.append(label)
    
    for idx in query_eval_4:
        img, label = dataset_4class[idx]
        qry_imgs.append(img)
        qry_lbls.append(label)
    
    sup_imgs = torch.stack(sup_imgs).to(device)
    sup_lbls = torch.tensor(sup_lbls).to(device)
    qry_imgs = torch.stack(qry_imgs).to(device)
    qry_lbls = torch.tensor(qry_lbls).to(device)
    
    # Within-domain evaluation
    acc_within, conf_within, _ = evaluate_prototypical_network(
        model_4class, sup_imgs, sup_lbls, qry_imgs, qry_lbls, n_way_4
    )
    
    # Cross-domain evaluation
    class_mapping = {0: 0, 1: 2, 2: 4, 3: 6}  # BF->BF, GF->GF, N->N, TF->TF
    
    acc_cross, conf_cross, preds_cross, true_cross = evaluate_cross_domain(
        model_7class, dataset_4class, support_eval_4, query_eval_4,
        n_way_4, device, class_mapping
    )
    
    print(f"\nWithin-Domain Accuracy (4-class trained & tested): {acc_within:.4f}")
    print(f"Cross-Domain Accuracy (7-class trained, 4-class tested): {acc_cross:.4f}")
    print(f"Performance Drop: {(acc_within - acc_cross)*100:.2f}%")
    
    # ========================================================================
    # PART 4: Statistical validation (30 episodes)
    # ========================================================================
    
    print("\n" + "="*80)
    print("PART 4: STATISTICAL VALIDATION (30 EPISODES)")
    print("="*80)
    
    n_runs = 30
    within_accs = []
    cross_accs = []
    
    for run in range(n_runs):
        sup_idx, qry_idx = split_support_query(dataset_4class, n_way_4, k_shot, query_size)
        
        s_imgs, s_lbls = [], []
        q_imgs, q_lbls = [], []
        
        for idx in sup_idx:
            img, label = dataset_4class[idx]
            s_imgs.append(img)
            s_lbls.append(label)
        
        for idx in qry_idx:
            img, label = dataset_4class[idx]
            q_imgs.append(img)
            q_lbls.append(label)
        
        s_imgs = torch.stack(s_imgs).to(device)
        s_lbls = torch.tensor(s_lbls).to(device)
        q_imgs = torch.stack(q_imgs).to(device)
        q_lbls = torch.tensor(q_lbls).to(device)
        
        # Within
        model_4class.eval()
        with torch.no_grad():
            dists = model_4class(s_imgs, s_lbls, q_imgs, n_way_4)
            preds = torch.argmin(dists, dim=1).cpu().numpy()
            within_accs.append(accuracy_score(q_lbls.cpu().numpy(), preds))
        
        # Cross
        acc_c, _, _, _ = evaluate_cross_domain(
            model_7class, dataset_4class, sup_idx, qry_idx, n_way_4, device, class_mapping
        )
        cross_accs.append(acc_c)
        
        if (run + 1) % 10 == 0:
            print(f"Episode {run+1}/{n_runs} complete")
    
    within_accs = np.array(within_accs)
    cross_accs = np.array(cross_accs)
    
    within_mean = np.mean(within_accs)
    within_std = np.std(within_accs)
    cross_mean = np.mean(cross_accs)
    cross_std = np.std(cross_accs)
    
    t_stat, p_value = stats.ttest_rel(within_accs, cross_accs)
    
    print(f"\nWithin-Domain:  {within_mean:.4f} ± {within_std:.4f}")
    print(f"Cross-Domain:   {cross_mean:.4f} ± {cross_std:.4f}")
    print(f"Mean Difference: {(within_mean - cross_mean):.4f}")
    print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    
    # ========================================================================
    # PART 5: Visualizations
    # ========================================================================
    
    class_names_4 = ['BF', 'GF', 'N', 'TF']
    
    # Confusion matrices comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(conf_within, annot=True, fmt='d', cmap='Greens', ax=axes[0],
                xticklabels=class_names_4, yticklabels=class_names_4,
                annot_kws={"size": 11, "weight": "bold"})
    axes[0].set_title(f'Within-Domain (Acc: {acc_within:.3f})', fontweight='bold', fontsize=12)
    axes[0].set_xlabel("Predicted", fontweight='bold')
    axes[0].set_ylabel("True", fontweight='bold')
    
    sns.heatmap(conf_cross, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=class_names_4, yticklabels=class_names_4,
                annot_kws={"size": 11, "weight": "bold"})
    axes[1].set_title(f'Cross-Domain (Acc: {acc_cross:.3f})', fontweight='bold', fontsize=12)
    axes[1].set_xlabel("Predicted", fontweight='bold')
    axes[1].set_ylabel("True", fontweight='bold')
    
    plt.suptitle('Generalization Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Box plot
    fig, ax = plt.subplots(figsize=(8, 6))
    box_data = [within_accs, cross_accs]
    bp = ax.boxplot(box_data, labels=['Within-Domain', 'Cross-Domain'],
                    patch_artist=True, widths=0.6)
    
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e67e22')
    
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax.set_title(f'Accuracy Distribution Over {n_runs} Episodes', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('accuracy_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    results = {
        'within_domain_mean': float(within_mean),
        'within_domain_std': float(within_std),
        'cross_domain_mean': float(cross_mean),
        'cross_domain_std': float(cross_std),
        'performance_drop_percent': float((within_mean - cross_mean) * 100),
        'p_value': float(p_value),
        't_statistic': float(t_stat)
    }
    
    with open('cross_domain_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*80)
    print("CROSS-DOMAIN VALIDATION COMPLETE")
    print("="*80)
    print("Results saved to 'cross_domain_results.json'")
    print("Figures saved as PNG files")
    print("="*80)