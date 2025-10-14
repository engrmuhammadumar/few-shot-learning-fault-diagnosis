# Cross-Domain Validation for 1440 RPM Dataset
# Tests generalization to different spindle speed (660 RPM → 1440 RPM)

import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
import seaborn as sns
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
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
# UTILITY FUNCTIONS
# ============================================================================

def load_spectrogram_dataset(root_path):
    """Load converted spectrogram images for 1440 RPM"""
    class_folders = {
        'BF': 'BF1440_1',
        'GF': 'GF1440_1', 
        'N': 'N1440_1',
        'TF': 'TF1440_1'
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
        
        # Take 110 samples per class
        img_files = img_files[:110]
        
        for img_file in img_files:
            file_paths.append(os.path.join(class_path, img_file))
            labels.append(idx)
        
        print(f"Loaded {class_name}: {len(img_files)} samples")
    
    return file_paths, labels, classes, transform


def split_support_query(file_paths, labels, n_way, k_shot, query_size, seed=42):
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
# MAIN EXECUTION
# ============================================================================

print("="*80)
print("CROSS-DOMAIN VALIDATION: 1440 RPM Dataset")
print("Testing generalization to different spindle speed")
print("="*80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# Load pre-trained model (quick training on 7-class)
print("Loading and training 7-class model (50 epochs)...")
backbone = Backbone()
model = PrototypicalNetwork(backbone).to(device)

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

sup_7, qry_7 = split_support_query(
    [p for p in range(len(dataset_7class))],
    dataset_7class.labels,
    7, 5, 12, seed=42
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
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
    
    dists = model(sup_imgs, sup_lbls, qry_imgs, 7)
    log_probs = -dists.log_softmax(dim=1)
    loss = nn.CrossEntropyLoss()(log_probs, qry_lbls)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("✓ Model ready!\n")

# ============================================================================
# EVALUATE ON 1440 RPM DATASET
# ============================================================================

print("="*80)
print("Loading 1440 RPM spectrogram dataset...")
print("="*80)

file_paths, labels, classes, transform = load_spectrogram_dataset(r"F:\D8B2_spectrograms")
print(f"\nTotal samples: {len(file_paths)}")
print(f"Classes: {classes}\n")

# Class mapping
class_mapping = {0: 0, 1: 2, 2: 4, 3: 6}

# Evaluate
print("Evaluating cross-domain performance (1440 RPM)...")
sup_idx, qry_idx = split_support_query(file_paths, labels, 4, 5, 14, seed=42)

sup_imgs, sup_lbls_4 = load_batch(file_paths, labels, sup_idx, transform, device)
sup_lbls_7 = torch.tensor([class_mapping[l.item()] for l in sup_lbls_4]).to(device)
qry_imgs, qry_lbls_4 = load_batch(file_paths, labels, qry_idx, transform, device)

model.eval()
with torch.no_grad():
    qry_emb = model.backbone(qry_imgs)
    sup_emb = model.backbone(sup_imgs)
    
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

true_labels = qry_lbls_4.cpu().numpy()
embeddings = qry_emb.cpu().numpy()

# Calculate metrics
accuracy = accuracy_score(true_labels, preds)
conf_matrix = confusion_matrix(true_labels, preds)
class_report = classification_report(true_labels, preds, target_names=classes, digits=3)

print(f"\nAccuracy (1440 RPM): {accuracy:.4f}")
print("="*80)

# ============================================================================
# GENERATE PLOTS
# ============================================================================

# 1. Confusion Matrix
print("Generating Confusion Matrix...")
fig, ax = plt.subplots(figsize=(8, 7))

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes,
            annot_kws={"size": 14, "weight": "bold"},
            cbar_kws={'label': 'Number of Samples'},
            linewidths=1, linecolor='gray', ax=ax)

ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')

# Make tick labels bold
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, fontweight='bold', rotation=0)


plt.tight_layout()
plt.savefig('confusion_matrix_cross_domain.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix_cross_domain.png\n")
plt.show()

# 2. t-SNE
print("\nGenerating t-SNE plot (1440 RPM)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=20, max_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(8, 6))

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
markers = ['o', 's', '^', 'D']

for i, class_name in enumerate(classes):
    idx = true_labels == i
    ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], 
              c=colors[i], marker=markers[i], s=100, 
              label=class_name, alpha=0.8, edgecolors='black', linewidth=1)

ax.set_xlabel('Component 1', fontsize=12, fontweight='bold')
ax.set_ylabel('Component 2', fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='best', frameon=True)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('tsne_1440rpm.png', dpi=300, bbox_inches='tight')
print("✓ Saved: tsne_1440rpm.png")
plt.show()

# 3. Classification Report
print("\nGenerating Classification Report (1440 RPM)...")
report_dict = classification_report(true_labels, preds, target_names=classes, 
                                   digits=3, output_dict=True)

report_data = []
for class_name in classes:
    report_data.append({
        'Class': class_name,
        'Precision': f"{report_dict[class_name]['precision']:.3f}",
        'Recall': f"{report_dict[class_name]['recall']:.3f}",
        'F1-Score': f"{report_dict[class_name]['f1-score']:.3f}",
        'Support': int(report_dict[class_name]['support'])
    })

df = pd.DataFrame(report_data)

fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=df.values, colLabels=df.columns,
                cellLoc='center', loc='center',
                colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for i in range(len(df.columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white', fontsize=11)

for i in range(1, len(classes) + 1):
    for j in range(len(df.columns)):
        cell = table[(i, j)]
        cell.set_facecolor('#ecf0f1')
        cell.set_text_props(weight='bold')

plt.tight_layout()
plt.savefig('classification_report_1440rpm.png', dpi=300, bbox_inches='tight')
print("✓ Saved: classification_report_1440rpm.png")
plt.show()

# ============================================================================
# PRINT RESULTS
# ============================================================================

print("\n" + "="*80)
print("CLASSIFICATION REPORT (1440 RPM)")
print("="*80)
print(class_report)
print("="*80)

# Save results
results = {
    'spindle_speed': '1440 RPM',
    'accuracy': float(accuracy),
    'confusion_matrix': conf_matrix.tolist(),
    'per_class_metrics': {
        classes[i]: {
            'precision': float(report_dict[classes[i]]['precision']),
            'recall': float(report_dict[classes[i]]['recall']),
            'f1-score': float(report_dict[classes[i]]['f1-score'])
        } for i in range(4)
    }
}

with open('results_1440rpm.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\n✓ All plots generated successfully!")
print("\nGenerated Files:")
print("  1. confusion_matrix_1440rpm.png")
print("  2. tsne_1440rpm.png")
print("  3. classification_report_1440rpm.png")
print("  4. results_1440rpm.json")
print("="*80)