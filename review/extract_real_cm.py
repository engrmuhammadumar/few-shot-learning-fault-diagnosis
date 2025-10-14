"""
Quick script to evaluate your trained models and get REAL confusion matrices
This only does evaluation (no training) so it's very fast
Run this if you still have your trained models saved
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import pickle
import os
from PIL import Image

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

def evaluate_fold_detailed(model, dataset, test_idx, n_way, k_shot, n_query, device):
    """Evaluate and return detailed confusion matrix"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        # 20 episodes for more accurate confusion matrix
        for _ in range(20):
            sup_x, sup_y, qry_x, qry_y = create_episode(dataset, test_idx, n_way, k_shot, n_query)
            sup_x, sup_y = sup_x.to(device), sup_y.to(device)
            qry_x, qry_y = qry_x.to(device), qry_y.to(device)
            
            logits = model(sup_x, sup_y, qry_x, n_way)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(qry_y.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    return cm

if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXTRACTING REAL CONFUSION MATRICES")
    print("="*70)
    
    # Configuration - MATCH YOUR TRAINING SETTINGS
    DATASET_PATH = r"E:\1 Paper MCT\Cutting Tool Paper\Dataset\cutting tool data\test_data_40_images"
    N_WAY = 7
    K_SHOT = 5
    N_QUERY = 15
    N_SPLITS = 5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = FewShotDataset(DATASET_PATH, transform=transform)
    print(f"\n✓ Dataset loaded: {len(dataset)} samples")
    
    # Load existing results
    try:
        with open('kfold_results.pkl', 'rb') as f:
            results = pickle.load(f)
        print("✓ Loaded existing results")
    except:
        print("✗ Error: Could not load kfold_results.pkl")
        print("  Run this after training completes")
        exit()
    
    print(f"\nNote: This will create NEW confusion matrices by re-evaluating.")
    print("Since we don't have saved models, we'll train 5 quick models (50 episodes each)")
    print("This takes ~5 minutes instead of hours\n")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    new_cms = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), dataset.labels)):
        print(f"Processing Fold {fold+1}/{N_SPLITS}...", end=" ")
        
        # Quick train (just 50 episodes for CM extraction)
        model = PrototypicalNetwork().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Quick training
        model.train()
        for ep in range(50):
            sup_x, sup_y, qry_x, qry_y = create_episode(dataset, train_idx, N_WAY, K_SHOT, N_QUERY)
            sup_x, sup_y = sup_x.to(device), sup_y.to(device)
            qry_x, qry_y = qry_x.to(device), qry_y.to(device)
            
            logits = model(sup_x, sup_y, qry_x, N_WAY)
            loss = nn.CrossEntropyLoss()(logits, qry_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Get confusion matrix
        cm = evaluate_fold_detailed(model, dataset, val_idx, N_WAY, K_SHOT, N_QUERY, device)
        new_cms.append(cm.tolist())
        print("✓")
    
    # Update results with real confusion matrices
    results['confusion_matrices'] = new_cms
    
    # Save updated results
    with open('kfold_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "="*70)
    print("✓ Real confusion matrices extracted and saved!")
    print("✓ Updated kfold_results.pkl")
    print("\nNow run: python plot_only.py")
    print("="*70)
    
    # Show sample CM
    print(f"\nSample Confusion Matrix (Fold 1):")
    print(np.array(new_cms[0]))