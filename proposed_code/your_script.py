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
import warnings
warnings.filterwarnings('ignore')

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
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam

def visualize_gradcam(model, dataset, indices, class_names, num_samples=5, device='cuda'):
    """Visualize GradCAM++ for sample images - saves combined and individual class images"""
    target_layer = model.backbone.model.layer4[-1].conv2
    gradcam = GradCAMPlusPlus(model, target_layer)
    
    # Organize indices by class
    class_indices = {class_name: [] for class_name in class_names}
    for idx in indices:
        _, label, _ = dataset[idx]
        class_indices[class_names[label]].append(idx)
    
    # Create combined figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    plot_idx = 0
    for class_name in class_names:
        if plot_idx >= num_samples:
            break
        if len(class_indices[class_name]) > 0:
            idx = class_indices[class_name][0]
            img, label, img_path = dataset[idx]
            
            # Original image
            original_img = Image.open(img_path).convert('RGB')
            original_img = original_img.resize((224, 224))
            original_array = np.array(original_img).astype(np.float32) / 255.0
            
            # Prepare input
            input_tensor = img.unsqueeze(0).to(device)
            
            # Generate GradCAM++
            cam = gradcam.generate_cam(input_tensor, label)
            
            # Create heatmap overlay
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # Overlay
            overlay = cv2.addWeighted(original_array, 0.6, heatmap, 0.4, 0)
            overlay = np.clip(overlay, 0, 1)
            
            # Plot in combined figure
            axes[plot_idx, 0].imshow(original_array)
            axes[plot_idx, 0].set_title(f'Original Image\nTrue Class: {class_name}', 
                                  fontweight='bold', fontsize=11)
            axes[plot_idx, 0].axis('off')
            
            axes[plot_idx, 1].imshow(cam, cmap='jet')
            axes[plot_idx, 1].set_title('GradCAM++ Activation\nHeatmap', 
                                  fontweight='bold', fontsize=11)
            axes[plot_idx, 1].axis('off')
            
            axes[plot_idx, 2].imshow(overlay)
            axes[plot_idx, 2].set_title('Discriminative Regions\n(Overlay)', 
                                  fontweight='bold', fontsize=11)
            axes[plot_idx, 2].axis('off')
            
            # Save individual class figure
            fig_individual, axes_ind = plt.subplots(1, 3, figsize=(12, 4))
            
            axes_ind[0].imshow(original_array)
            axes_ind[0].set_title(f'Original Image\nClass: {class_name}', 
                                fontweight='bold', fontsize=12)
            axes_ind[0].axis('off')
            
            axes_ind[1].imshow(cam, cmap='jet')
            axes_ind[1].set_title('GradCAM++ Activation\nHeatmap', 
                                fontweight='bold', fontsize=12)
            axes_ind[1].axis('off')
            
            axes_ind[2].imshow(overlay)
            axes_ind[2].set_title('Discriminative Regions\n(Overlay)', 
                                fontweight='bold', fontsize=12)
            axes_ind[2].axis('off')
            
            plt.suptitle(f'GradCAM++ Interpretability: {class_name}', 
                        fontweight='bold', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'08_{plot_idx+1}_gradcam_{class_name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig_individual)
            print(f"✓ Saved: 08_{plot_idx+1}_gradcam_{class_name}.png")
            
            plot_idx += 1
    
    # Save combined figure
    plt.suptitle('GradCAM++ Interpretability: Signal Regions Contributing to Classification', 
                fontweight='bold', fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig('08_gradcam_all_classes_combined.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("✓ Saved: 08_gradcam_all_classes_combined.png")

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
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    
    for idx, class_name in enumerate(class_names):
        if class_cams[class_name]:
            avg_cam = np.mean(class_cams[class_name], axis=0)
            im = axes[idx].imshow(avg_cam, cmap='jet', vmin=0, vmax=1)
            axes[idx].set_title(f'{class_name}\nAverage Discriminative Pattern', 
                              fontweight='bold', fontsize=12)
            axes[idx].axis('off')
            cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=9)
    
    # Hide the last axis if odd number of classes
    if len(class_names) < len(axes):
        axes[-1].axis('off')
    
    plt.suptitle('Class-Specific Average Activation Maps: Most Important Signal Regions per Tool Condition', 
                fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig('09_class_average_activation_maps.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("✓ Saved: 09_class_average_activation_maps.png")

def generate_activation_statistics(model, dataset, indices, class_names, device='cuda'):
    """Generate quantitative statistics about activation patterns"""
    target_layer = model.backbone.model.layer4[-1].conv2
    gradcam = GradCAMPlusPlus(model, target_layer)
    
    class_stats = {}
    
    for class_name in class_names:
        class_stats[class_name] = {
            'mean_activation': [],
            'max_activation': [],
            'activation_variance': []
        }
    
    for idx in indices:
        img, label, img_path = dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)
        cam = gradcam.generate_cam(input_tensor, label)
        
        class_name = class_names[label]
        class_stats[class_name]['mean_activation'].append(np.mean(cam))
        class_stats[class_name]['max_activation'].append(np.max(cam))
        class_stats[class_name]['activation_variance'].append(np.var(cam))
    
    # Plot statistics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['mean_activation', 'max_activation', 'activation_variance']
    titles = ['Mean Activation Strength', 'Peak Activation Strength', 'Activation Variance']
    
    for ax_idx, (metric, title) in enumerate(zip(metrics, titles)):
        data = [class_stats[cls][metric] for cls in class_names]
        bp = axes[ax_idx].boxplot(data, labels=class_names, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[ax_idx].set_title(title, fontweight='bold', fontsize=12)
        axes[ax_idx].set_xlabel('Tool Condition', fontweight='bold', fontsize=11)
        axes[ax_idx].set_ylabel('Activation Value', fontweight='bold', fontsize=11)
        axes[ax_idx].grid(True, alpha=0.3, axis='y')
        axes[ax_idx].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Quantitative Analysis of Discriminative Region Activation Patterns', 
                fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig('10_activation_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("✓ Saved: 10_activation_statistics.png")

def visualize_correct_vs_incorrect(model, dataset, support_images, support_labels, 
                                  query_indices, query_labels, class_names, n_way, device='cuda'):
    """Compare GradCAM++ for correctly and incorrectly classified samples"""
    model.eval()
    
    # Get predictions
    query_images_list = []
    query_paths = []
    for idx in query_indices:
        img, _, path = dataset[idx]
        query_images_list.append(img)
        query_paths.append(path)
    
    query_images = torch.stack(query_images_list).to(device)
    
    with torch.no_grad():
        dists = model(support_images, support_labels, query_images, n_way)
        preds = torch.argmin(dists, dim=1).cpu().numpy()
    
    true_labels = query_labels.cpu().numpy()
    
    # Find correct and incorrect predictions
    correct_mask = preds == true_labels
    incorrect_mask = ~correct_mask
    
    if np.sum(incorrect_mask) > 0:
        # Visualize incorrect predictions
        incorrect_indices = np.where(incorrect_mask)[0][:3]  # Up to 3 examples
        
        target_layer = model.backbone.model.layer4[-1].conv2
        gradcam = GradCAMPlusPlus(model, target_layer)
        
        fig, axes = plt.subplots(len(incorrect_indices), 3, figsize=(12, 4*len(incorrect_indices)))
        if len(incorrect_indices) == 1:
            axes = axes.reshape(1, -1)
        
        for plot_idx, idx in enumerate(incorrect_indices):
            img = query_images_list[idx]
            true_label = true_labels[idx]
            pred_label = preds[idx]
            
            original_img = Image.open(query_paths[idx]).convert('RGB').resize((224, 224))
            original_array = np.array(original_img).astype(np.float32) / 255.0
            
            input_tensor = img.unsqueeze(0).to(device)
            cam = gradcam.generate_cam(input_tensor, pred_label)
            
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            overlay = np.clip(cv2.addWeighted(original_array, 0.6, heatmap, 0.4, 0), 0, 1)
            
            axes[plot_idx, 0].imshow(original_array)
            axes[plot_idx, 0].set_title(f'Original\nTrue: {class_names[true_label]}', 
                                       fontweight='bold', color='red')
            axes[plot_idx, 0].axis('off')
            
            axes[plot_idx, 1].imshow(cam, cmap='jet')
            axes[plot_idx, 1].set_title(f'Focus Regions\nPredicted: {class_names[pred_label]}', 
                                       fontweight='bold', color='red')
            axes[plot_idx, 1].axis('off')
            
            axes[plot_idx, 2].imshow(overlay)
            axes[plot_idx, 2].set_title('Misclassification Pattern', 
                                       fontweight='bold', color='red')
            axes[plot_idx, 2].axis('off')
        
        plt.suptitle('Error Analysis: Discriminative Regions in Misclassified Samples', 
                    fontweight='bold', fontsize=14, color='red')
        plt.tight_layout()
        plt.savefig('11_misclassification_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        print("✓ Saved: 11_misclassification_analysis.png")
    else:
        print("✓ Perfect classification - no misclassification analysis needed!")

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
        img_display = img.permute(1, 2, 0).numpy()
        img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_display = np.clip(img_display, 0, 1)
        axs[i].imshow(img_display)
        axs[i].set_title(f"Class: {dataset.classes[label]}", fontweight='bold')
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig('01_sample_images.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("✓ Saved: 01_sample_images.png")

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
    
    # Plot 1: Loss
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, loss_history, '-', linewidth=2, color='#2E86AB')
    ax.set_xlabel('Episode', fontweight='bold', fontsize=12)
    ax.set_ylabel('Loss', fontweight='bold', fontsize=12)
    ax.set_title('Training Loss Convergence', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('02_training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("✓ Saved: 02_training_loss.png")
    
    # Plot 2: Accuracy
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, accuracy_history, '-', linewidth=2, color='#A23B72')
    ax.set_xlabel('Episode', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax.set_title('Classification Accuracy', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('03_training_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("✓ Saved: 03_training_accuracy.png")
    
    # Plot 3: Combined
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, loss_history, '-', linewidth=2, color='#2E86AB')
    ax1.set_xlabel('Episode', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Loss', fontweight='bold', fontsize=12)
    ax1.set_title('Training Loss Convergence', fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, accuracy_history, '-', linewidth=2, color='#A23B72')
    ax2.set_xlabel('Episode', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax2.set_title('Classification Accuracy', fontweight='bold', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('04_training_progress_combined.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("✓ Saved: 04_training_progress_combined.png")

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
    
    # Plot t-SNE for test data
    plt.figure(figsize=(7, 5))
    for i in np.unique(labels):  # Unique labels in the test set
        idx = labels == i
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=class_names[i], alpha=1)
    
    plt.legend()
    plt.xlabel('Component 1', fontweight='bold')
    plt.ylabel('Component 2', fontweight='bold')
    plt.tight_layout()
    plt.savefig('06_tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("✓ Saved: 06_tsne_visualization.png")

def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(9, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12, "weight": "bold"}, cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted Label", fontweight='bold', fontsize=13)
    plt.ylabel("True Label", fontweight='bold', fontsize=13)
    plt.title('Confusion Matrix', fontweight='bold', fontsize=14)
    plt.xticks(fontsize=11, fontweight='bold', rotation=45)
    plt.yticks(fontsize=11, fontweight='bold', rotation=0)
    plt.tight_layout()
    plt.savefig('05_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("✓ Saved: 05_confusion_matrix.png")

def plot_roc_curves(y_true, probs, class_names, n_way):
    y_true_bin = label_binarize(y_true, classes=list(range(n_way)))
    
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_way):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(11, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, n_way))
    
    for i, color in zip(range(n_way), colors):
        plt.plot(fpr[i], tpr[i], lw=2.5, color=color,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier', alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=13)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=13)
    plt.title('ROC Curves - Multi-class Classification Performance', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right", fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('07_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("✓ Saved: 07_roc_curves.png")

def save_model(model, optimizer, loss_history, accuracy_history, filepath='prototypical_model.pth'):
    """Save model and training history"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'accuracy_history': accuracy_history,
    }, filepath)
    print(f"✓ Model saved to: {filepath}")

def load_model(model, optimizer, filepath='prototypical_model.pth'):
    """Load model and training history"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_history = checkpoint['loss_history']
        accuracy_history = checkpoint['accuracy_history']
        print(f"✓ Model loaded from: {filepath}")
        return loss_history, accuracy_history
    else:
        return None, None

# ======================= Main Execution =======================
def main():
    print("\n" + "="*80)
    print("PROTOTYPICAL NETWORK WITH GRADCAM++ INTERPRETABILITY ANALYSIS")
    print("="*80 + "\n")
    
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
    
    print(f"✓ Dataset loaded: {len(dataset)} samples, {len(dataset.classes)} classes")
    print(f"✓ Classes: {class_names}\n")
    
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
    
    # Try to load existing model
    model_path = 'prototypical_model.pth'
    loss_history, accuracy_history = load_model(model, optimizer, model_path)
    
    if loss_history is None:
        # Train new model
        print("No saved model found. Training new model...")
        print("This will take a while (500 epochs)...\n")
        
        loss_history, accuracy_history = train_prototypical_network(
            model, optimizer, dataset, support_indices, query_indices,
            n_way, k_shot, query_size, num_epochs, device
        )
        print("\n✓ Training completed!")
        
        # Save the trained model
        save_model(model, optimizer, loss_history, accuracy_history, model_path)
    else:
        print("✓ Using previously trained model (skipping training)\n")
    
    # Visualize samples
    print("Generating sample visualizations...")
    visualize_samples(dataset)
    
    # Plot training progress
    plot_training_progress(loss_history, accuracy_history)
    
    # Evaluation
    print("\nPreparing evaluation data...")
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
    print("Evaluating model performance...\n")
    accuracy_eval, conf_matrix_eval, class_report_eval, preds, dists = evaluate_prototypical_network(
        model, support_images_eval, support_labels_eval, 
        query_images_eval, query_labels_eval, n_way
    )
    
    print(f"Evaluation Accuracy: {accuracy_eval:.4f}\n")
    print("Classification Report:\n", class_report_eval)
    
    # Confusion Matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(conf_matrix_eval, class_names)
    
    # t-SNE Visualization
    print("Generating t-SNE visualization...")
    query_embeddings_eval = extract_embeddings(model, query_images_eval)
    plot_tsne(query_embeddings_eval, query_labels_eval.cpu().numpy(), class_names)
    
    # ROC Curves
    print("Generating ROC curves...")
    with torch.no_grad():
        log_probs = -dists.log_softmax(dim=1)
        probs = log_probs.exp().cpu().numpy()
    
    plot_roc_curves(query_labels_eval.cpu().numpy(), probs, class_names, n_way)
    
    # ======================= INTERPRETABILITY ANALYSIS =======================
    print("\n" + "="*80)
    print("INTERPRETABILITY ANALYSIS - GradCAM++")
    print("Addressing Reviewer Concern: Which signal regions contribute to decisions?")
    print("="*80 + "\n")
    
    # 1. Visualize GradCAM++ for individual samples
    print("1. Generating GradCAM++ visualizations for query samples...")
    visualize_gradcam(model, dataset, query_eval_indices, class_names, 
                     num_samples=7, device=device)
    
    # 2. Analyze average feature importance per class
    print("\n2. Analyzing average activation patterns per class...")
    analyze_feature_importance(model, dataset, query_eval_indices, class_names, device)
    
    # 3. Quantitative activation statistics
    print("\n3. Computing quantitative activation statistics...")
    generate_activation_statistics(model, dataset, query_eval_indices, class_names, device)
    
    # 4. Error analysis (if any misclassifications)
    print("\n4. Performing error analysis on misclassifications...")
    visualize_correct_vs_incorrect(model, dataset, support_images_eval, support_labels_eval,
                                   query_eval_indices, query_labels_eval, class_names, n_way, device)
    
    print("\n" + "="*80)
    print("✓ ALL ANALYSES COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files (numbered for easy reference):")
    print("─" * 80)
    print("Model Checkpoint:")
    print("  • prototypical_model.pth - Trained model (reusable)")
    print("\nStandard Evaluation Plots:")
    print("  01_sample_images.png - Sample dataset images")
    print("  02_training_loss.png - Training loss curve (separate)")
    print("  03_training_accuracy.png - Training accuracy curve (separate)")
    print("  04_training_progress_combined.png - Loss + Accuracy (combined)")
    print("  05_confusion_matrix.png - Classification confusion matrix")
    print("  06_tsne_visualization.png - t-SNE embedding visualization")
    print("  07_roc_curves.png - ROC curves for all classes")
    print("\nInterpretability Analysis (Answers Reviewer's Question):")
    print("  08_1_gradcam_BF.png - GradCAM++ for BF class")
    print("  08_2_gradcam_BFI.png - GradCAM++ for BFI class")
    print("  08_3_gradcam_GF.png - GradCAM++ for GF class")
    print("  08_4_gradcam_GFI.png - GradCAM++ for GFI class")
    print("  08_5_gradcam_N.png - GradCAM++ for N class")
    print("  08_6_gradcam_NI.png - GradCAM++ for NI class")
    print("  08_7_gradcam_TF.png - GradCAM++ for TF class")
    print("  08_gradcam_all_classes_combined.png - All GradCAM++ combined")
    print("  09_class_average_activation_maps.png - Average patterns per class")
    print("  10_activation_statistics.png - Quantitative activation analysis")
    print("  11_misclassification_analysis.png - Error patterns (if applicable)")
    print("─" * 80)
    print("\n✓ All plots saved separately with sequential numbering!")
    print("✓ GradCAM++ saved for EACH class individually (7 separate files)!")
    print("✓ t-SNE visualization updated to your preferred style!")
    print("✓ These visualizations demonstrate which specific signal regions")
    print("  contribute most to the model's classification decisions!")
    print("\n✓ To regenerate interpretability plots without retraining:")
    print("  Just run this script again - it will load the saved model!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()