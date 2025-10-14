# Core Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

# Utilities
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import os
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split

# ======================= Dataset Class =======================
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
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
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

        return image, label, img_path

# ======================= GradCAM++ Implementation =======================
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class):
        self.model.eval()
        
        # Forward pass
        model_output = self.model.backbone(input_image)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_loss = model_output[0, target_class]
        class_loss.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        # Calculate weights using GradCAM++ formula
        alpha_num = gradients ** 2
        alpha_denom = 2 * (gradients ** 2) + \
                      np.sum(activations * (gradients ** 3), axis=(1, 2), keepdims=True)
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1.0)
        alphas = alpha_num / alpha_denom
        
        weights = np.sum(alphas * np.maximum(gradients, 0), axis=(1, 2))
        
        # Generate CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

def visualize_gradcam(model, dataset, indices, class_names, num_samples=5, device='cuda'):
    """Visualize GradCAM++ for sample images"""
    # Get the target layer (last convolutional layer of ResNet18)
    target_layer = model.backbone.model.layer4[-1].conv2
    gradcam = GradCAMPlusPlus(model, target_layer)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(min(num_samples, len(indices))):
        img, label, img_path = dataset[indices[idx]]
        
        # Original image
        original_img = Image.open(img_path).convert('RGB')
        original_img = original_img.resize((224, 224))
        
        # Prepare input
        input_tensor = img.unsqueeze(0).to(device)
        
        # Generate GradCAM++
        cam = gradcam.generate_cam(input_tensor, label)
        
        # Create heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        original_array = np.array(original_img)
        overlay = cv2.addWeighted(original_array, 0.6, heatmap, 0.4, 0)
        
        # Plot
        axes[idx, 0].imshow(original_img)
        axes[idx, 0].set_title(f'Original\nClass: {class_names[label]}', fontweight='bold')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(cam, cmap='jet')
        axes[idx, 1].set_title('GradCAM++ Heatmap', fontweight='bold')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title('Overlay', fontweight='bold')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_importance(model, dataset, indices, class_names, device='cuda'):
    """Analyze which regions are most important for each class"""
    target_layer = model.backbone.model.layer4[-1].conv2
    gradcam = GradCAMPlusPlus(model, target_layer)
    
    class_cams = {class_name: [] for class_name in class_names}
    
    for idx in indices:
        img, label, img_path = dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)
        
        cam = gradcam.generate_cam(input_tensor, label)
        class_cams[class_names[label]].append(cam)
    
    # Calculate average activation maps per class
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, class_name in enumerate(class_names):
        if class_cams[class_name]:
            avg_cam = np.mean(class_cams[class_name], axis=0)
            im = axes[idx].imshow(avg_cam, cmap='jet')
            axes[idx].set_title(f'{class_name}\nAverage Activation', fontweight='bold')
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046)
    
    # Hide the last axis if odd number of classes
    if len(class_names) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('class_average_activation_maps.png', dpi=300, bbox_inches='tight')
    plt.show()

# ======================= Model Classes =======================
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

# ======================= Helper Functions =======================
def split_support_query(dataset, n_way, k_shot, query_size):
    class_to_indices = {cls: [] for cls in range(len(dataset.classes))}
    for idx in range(len(dataset)):
        _, label, _ = dataset[idx]
        class_to_indices[label].append(idx)
    
    support_set = []
    query_set = []
    for cls in range(n_way):
        indices = class_to_indices[cls]
        support_indices, query_indices = train_test_split(
            indices, train_size=k_shot, test_size=query_size, random_state=42
        )
        support_set.extend(support_indices)
        query_set.extend(query_indices)
    
    return support_set, query_set

def visualize_samples(dataset, num_samples=5):
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        img, label, _ = dataset[i]
        axs[i].imshow(img.permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axs[i].set_title(f"Class: {dataset.classes[label]}", fontweight='bold')
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_prototypical_network(model, optimizer, dataset, support_indices, query_indices, 
                               n_way, k_shot, query_size, num_epochs, device):
    model.to(device)
    loss_history = []
    accuracy_history = []
    
    for epoch in range(num_epochs):
        model.train()
        
        support_images, support_labels = [], []
        query_images, query_labels = [], []
        
        for idx in support_indices:
            img, label, _ = dataset[idx]
            support_images.append(img)
            support_labels.append(label)
        
        for idx in query_indices:
            img, label, _ = dataset[idx]
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
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
    
    return loss_history, accuracy_history

def plot_training_progress(loss_history, accuracy_history):
    epochs = range(1, len(loss_history) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, loss_history, '-', linewidth=2, color='#2E86AB')
    ax1.set_xlabel('Episode', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Loss', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, accuracy_history, '-', linewidth=2, color='#A23B72')
    ax2.set_xlabel('Episode', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

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
            target_names=[f"Class {i}" for i in range(n_way)]
        )
        
    return accuracy, conf_matrix, class_report, preds, dists

def extract_embeddings(model, images):
    model.eval()
    with torch.no_grad():
        embeddings = model.backbone(images)
    return embeddings.cpu().numpy()

def plot_tsne(embeddings, labels, class_names):
    tsne = TSNE(n_components=2, perplexity=40, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    for i in np.unique(labels):
        idx = labels == i
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], 
                   label=class_names[i], alpha=0.7, s=50)
    
    plt.legend(fontsize=10)
    plt.xlabel('Component 1', fontweight='bold', fontsize=12)
    plt.ylabel('Component 2', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(8, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 11, "weight": "bold"})
    plt.xlabel("Predicted Label", fontweight='bold', fontsize=12)
    plt.ylabel("True Label", fontweight='bold', fontsize=12)
    plt.xticks(fontsize=10, fontweight='bold', rotation=45)
    plt.yticks(fontsize=10, fontweight='bold', rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curves(y_true, probs, class_names, n_way):
    y_true_bin = label_binarize(y_true, classes=list(range(n_way)))
    
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_way):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_way))
    
    for i, color in zip(range(n_way), colors):
        plt.plot(fpr[i], tpr[i], lw=2.5, color=color,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# ======================= Main Execution =======================
def main():
    # Data transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset_path = r"E:\1 Paper MCT\Cutting Tool Paper\Dataset\cutting tool data\test_data_40_images"
    dataset = FewShotDataset(dataset_path, transform=transform)
    class_names = ['BF', 'BFI', 'GF', 'GFI', 'N', 'NI', 'TF']
    
    print(f"Dataset loaded: {len(dataset)} samples, {len(dataset.classes)} classes")
    print(f"Classes: {class_names}\n")
    
    # Visualize samples
    visualize_samples(dataset)
    
    # Parameters
    n_way = 7
    k_shot = 5
    query_size = 15
    num_epochs = 500
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Split dataset
    support_indices, query_indices = split_support_query(dataset, n_way, k_shot, query_size)
    print(f"Support set: {len(support_indices)}, Query set: {len(query_indices)}\n")
    
    # Initialize model
    backbone = Backbone()
    model = PrototypicalNetwork(backbone).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train model
    print("Training started...")
    loss_history, accuracy_history = train_prototypical_network(
        model, optimizer, dataset, support_indices, query_indices,
        n_way, k_shot, query_size, num_epochs, device
    )
    print("\nTraining completed!\n")
    
    # Plot training progress
    plot_training_progress(loss_history, accuracy_history)
    
    # Evaluation
    print("Preparing evaluation data...")
    support_eval_indices, query_eval_indices = split_support_query(
        dataset, n_way, k_shot, query_size
    )
    
    support_images_eval, support_labels_eval = [], []
    query_images_eval, query_labels_eval = [], []
    
    for idx in support_eval_indices:
        img, label, _ = dataset[idx]
        support_images_eval.append(img)
        support_labels_eval.append(label)
    
    for idx in query_eval_indices:
        img, label, _ = dataset[idx]
        query_images_eval.append(img)
        query_labels_eval.append(label)
    
    support_images_eval = torch.stack(support_images_eval).to(device)
    support_labels_eval = torch.tensor(support_labels_eval).to(device)
    query_images_eval = torch.stack(query_images_eval).to(device)
    query_labels_eval = torch.tensor(query_labels_eval).to(device)
    
    # Evaluate
    accuracy_eval, conf_matrix_eval, class_report_eval, preds, dists = evaluate_prototypical_network(
        model, support_images_eval, support_labels_eval, 
        query_images_eval, query_labels_eval, n_way
    )
    
    print(f"Evaluation Accuracy: {accuracy_eval:.4f}\n")
    print("Classification Report:\n", class_report_eval)
    
    # Confusion Matrix
    plot_confusion_matrix(conf_matrix_eval, class_names)
    
    # t-SNE Visualization
    print("\nGenerating t-SNE visualization...")
    query_embeddings_eval = extract_embeddings(model, query_images_eval)
    plot_tsne(query_embeddings_eval, query_labels_eval.cpu().numpy(), class_names)
    
    # ROC Curves
    print("Generating ROC curves...")
    with torch.no_grad():
        log_probs = -dists.log_softmax(dim=1)
        probs = log_probs.exp().cpu().numpy()
    
    plot_roc_curves(query_labels_eval.cpu().numpy(), probs, class_names, n_way)
    
    # ======================= INTERPRETABILITY ANALYSIS =======================
    print("\n" + "="*70)
    print("INTERPRETABILITY ANALYSIS - GradCAM++")
    print("="*70 + "\n")
    
    # 1. Visualize GradCAM++ for individual samples
    print("Generating GradCAM++ visualizations for query samples...")
    visualize_gradcam(model, dataset, query_eval_indices, class_names, 
                     num_samples=7, device=device)
    
    # 2. Analyze average feature importance per class
    print("Analyzing average activation patterns per class...")
    analyze_feature_importance(model, dataset, query_eval_indices, class_names, device)
    
    print("\n" + "="*70)
    print("All visualizations saved successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("1. sample_images.png - Sample dataset images")
    print("2. training_progress.png - Training loss and accuracy")
    print("3. confusion_matrix.png - Classification confusion matrix")
    print("4. tsne_visualization.png - t-SNE embedding visualization")
    print("5. roc_curves.png - ROC curves for all classes")
    print("6. gradcam_visualization.png - GradCAM++ for individual samples")
    print("7. class_average_activation_maps.png - Average activation per class")

if __name__ == "__main__":
    main()