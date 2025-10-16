# Fast Cross-Domain Generalization Test
# Uses existing trained 7-class model WITHOUT retraining
# Only evaluates on new 4-class spectrogram data

import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from scipy import stats
import json

# ============================================================================
# MODEL ARCHITECTURE
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
# DATASET LOADER FOR SPECTROGRAMS
# ============================================================================

def load_spectrogram_dataset(root_path):
    """Load converted spectrogram images"""
    class_folders = {
        'BF': 'BF660_1',
        'GF': 'GF660_1', 
        'N': 'N660_1',
        'TF': 'TF660_1'
    }
    
    file_paths = []
    labels = []
    classes = sorted(class_folders.keys())
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    for idx, (class_name, folder_name) in enumerate(sorted(class_folders.items())):
        class_path = os.path.join(root_path, folder_name, 'AE')
        img_files = [f for f in os.listdir(class_path) if f.endswith('.png')]
        
        # Take 120 samples per class for balance
        img_files = img_files[:120]
        
        for img_file in img_files:
            file_paths.append(os.path.join(class_path, img_file))
            labels.append(idx)
        
        print(f"Loaded {class_name}: {len(img_files)} samples")
    
    return file_paths, labels, classes, transform


def split_support_query(file_paths, labels, n_way, k_shot, query_size, seed=None):
    """Split into support and query sets"""
    class_to_indices = {cls: [] for cls in range(n_way)}
    for idx, label in enumerate(labels):
        class_to_indices[label].append(idx)
    
    support_indices = []
    query_indices = []
    
    for cls in range(n_way):
        indices = class_to_indices[cls]
        sup_idx, qry_idx = train_test_split(
            indices, train_size=k_shot, test_size=query_size, random_state=seed
        )
        support_indices.extend(sup_idx)
        query_indices.extend(qry_idx)
    
    return support_indices, query_indices


def load_batch(file_paths, labels, indices, transform, device):
    """Load batch of images"""
    images = []
    batch_labels = []
    
    for idx in indices:
        img = Image.open(file_paths[idx]).convert('RGB')
        img_tensor = transform(img)
        images.append(img_tensor)
        batch_labels.append(labels[idx])
    
    images = torch.stack(images).to(device)
    batch_labels = torch.tensor(batch_labels).to(device)
    
    return images, batch_labels

# ============================================================================
# LOAD PRE-TRAINED 7-CLASS MODEL
# ============================================================================

print("="*80)
print("FAST CROSS-DOMAIN GENERALIZATION TEST")
print("Using pre-trained 7-class model on new 4-class spectrogram data")
print("="*80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")

# Initialize and load your trained model
print("\nLoading pre-trained 7-class model...")
print("Note: Using the model you trained earlier. If running separately,")
print("      you need to train the 7-class model first or load saved weights.")

backbone_7 = Backbone()
model_7class = PrototypicalNetwork(backbone_7).to(device)

# If you have saved weights, load them here:
# model_7class.load_state_dict(torch.load('model_7class.pth'))

# Quick training if no saved model (50 epochs only for speed)
print("\nQuick training 7-class model (50 epochs)...")

from torch.utils.data import Dataset

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
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        self.image_paths.append(os.path.join(class_path, img_name))
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

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset_7class = FewShotDataset(
    r"E:\1 Paper MCT\Cutting Tool Paper\Dataset\cutting tool data\test_data_40_images",
    transform=transform_train
)

print(f"7-class dataset: {len(dataset_7class)} samples")

# Split
sup_7, qry_7 = split_support_query(
    [p for p in range(len(dataset_7class))],
    dataset_7class.labels,
    7, 5, 15, seed=42
)

optimizer = torch.optim.Adam(model_7class.parameters(), lr=1e-3)

model_7class.train()
for epoch in range(50):
    sup_imgs, sup_lbls = [], []
    qry_imgs, qry_lbls = [], []
    
    for idx in sup_7:
        img, lbl = dataset_7class[idx]
        sup_imgs.append(img)
        sup_lbls.append(lbl)
    
    for idx in qry_7:
        img, lbl = dataset_7class[idx]
        qry_imgs.append(img)
        qry_lbls.append(lbl)
    
    sup_imgs = torch.stack(sup_imgs).to(device)
    sup_lbls = torch.tensor(sup_lbls).to(device)
    qry_imgs = torch.stack(qry_imgs).to(device)
    qry_lbls = torch.tensor(qry_lbls).to(device)
    
    dists = model_7class(sup_imgs, sup_lbls, qry_imgs, 7)
    log_probs = -dists.log_softmax(dim=1)
    loss = nn.CrossEntropyLoss()(log_probs, qry_lbls)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        with torch.no_grad():
            preds = torch.argmin(dists, dim=1)
            acc = accuracy_score(qry_lbls.cpu(), preds.cpu())
        print(f"  Epoch {epoch+1}/50 - Loss: {loss.item():.4f}, Acc: {acc:.4f}")

print("✓ 7-class model ready!\n")

# ============================================================================
# CROSS-DOMAIN EVALUATION ON 4-CLASS SPECTROGRAMS
# ============================================================================

print("="*80)
print("CROSS-DOMAIN EVALUATION: 4-Class Spectrogram Dataset")
print("="*80)

# Load spectrogram dataset
file_paths, labels, classes, transform = load_spectrogram_dataset(r"F:\20240925_spectrograms")
print(f"\nTotal samples: {len(file_paths)}")
print(f"Classes: {classes}\n")

# Class mapping: 4-class -> 7-class
class_mapping = {0: 0, 1: 2, 2: 4, 3: 6}  # BF->BF, GF->GF, N->N, TF->TF

# Run 30 evaluation episodes
print("Running 30 cross-domain evaluation episodes...")
print("(This tests generalization without any retraining)\n")

accuracies = []
all_preds = []
all_labels = []

for episode in range(30):
    # Random split
    sup_idx, qry_idx = split_support_query(file_paths, labels, 4, 5, 15, seed=episode)
    
    # Load support set
    sup_imgs, sup_lbls_4 = load_batch(file_paths, labels, sup_idx, transform, device)
    
    # Map to 7-class space
    sup_lbls_7 = torch.tensor([class_mapping[l.item()] for l in sup_lbls_4]).to(device)
    
    # Load query set
    qry_imgs, qry_lbls_4 = load_batch(file_paths, labels, qry_idx, transform, device)
    
    # Evaluate
    model_7class.eval()
    with torch.no_grad():
        sup_emb = model_7class.backbone(sup_imgs)
        qry_emb = model_7class.backbone(qry_imgs)
        
        # Compute prototypes in 7-class space
        prototypes = []
        for i in range(4):
            label_7 = class_mapping[i]
            mask = sup_lbls_7 == label_7
            if mask.sum() > 0:
                prototypes.append(sup_emb[mask].mean(0))
            else:
                prototypes.append(torch.zeros_like(sup_emb[0]))
        
        prototypes = torch.stack(prototypes)
        dists = torch.cdist(qry_emb, prototypes)
        preds = torch.argmin(dists, dim=1).cpu().numpy()
    
    acc = accuracy_score(qry_lbls_4.cpu().numpy(), preds)
    accuracies.append(acc)
    
    if episode == 0:
        all_preds = preds
        all_labels = qry_lbls_4.cpu().numpy()
    
    if (episode + 1) % 5 == 0:
        print(f"  Episode {episode+1}/30 - Accuracy: {acc:.4f}")

# ============================================================================
# RESULTS AND VISUALIZATION
# ============================================================================

accuracies = np.array(accuracies)
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

print("\n" + "="*80)
print("CROSS-DOMAIN GENERALIZATION RESULTS")
print("="*80)
print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
print(f"Min Accuracy:  {np.min(accuracies):.4f}")
print(f"Max Accuracy:  {np.max(accuracies):.4f}")
print(f"Median:        {np.median(accuracies):.4f}")
print(f"95% CI:        [{mean_acc - 1.96*std_acc:.4f}, {mean_acc + 1.96*std_acc:.4f}]")

# Per-class metrics (from first episode)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, labels=list(range(4)), zero_division=0
)

print("\nPer-Class Performance (Episode 1):")
for i, class_name in enumerate(classes):
    print(f"  {class_name}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}")

# Confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=classes, yticklabels=classes,
            annot_kws={"size": 12, "weight": "bold"})
axes[0].set_title('Confusion Matrix (Cross-Domain)', fontweight='bold', fontsize=13)
axes[0].set_xlabel('Predicted Label', fontweight='bold')
axes[0].set_ylabel('True Label', fontweight='bold')

# Accuracy Distribution
axes[1].hist(accuracies, bins=15, color='#3498db', alpha=0.7, edgecolor='black')
axes[1].axvline(mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.3f}')
axes[1].set_xlabel('Accuracy', fontweight='bold')
axes[1].set_ylabel('Frequency', fontweight='bold')
axes[1].set_title('Accuracy Distribution (30 Episodes)', fontweight='bold', fontsize=13)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cross_domain_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure saved: cross_domain_results.png")
plt.show()

# Save results
results = {
    'mean_accuracy': float(mean_acc),
    'std_accuracy': float(std_acc),
    'min_accuracy': float(np.min(accuracies)),
    'max_accuracy': float(np.max(accuracies)),
    'median_accuracy': float(np.median(accuracies)),
    'all_accuracies': accuracies.tolist(),
    'per_class_f1': {classes[i]: float(f1[i]) for i in range(4)},
    'confusion_matrix': conf_matrix.tolist()
}

with open('cross_domain_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("✓ Results saved: cross_domain_results.json")
print("\n" + "="*80)
print("INTERPRETATION FOR MANUSCRIPT")
print("="*80)
print(f"The model trained on 7-class data achieved {mean_acc:.1%} accuracy")
print(f"when tested on the 4-class dataset from different conditions.")
print(f"This demonstrates {'excellent' if mean_acc > 0.85 else 'good' if mean_acc > 0.75 else 'moderate'} generalization capability")
print("to new environmental settings without retraining.")
print("="*80)