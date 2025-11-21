import os
import sys
# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc
from model.msan import MSAN
from data.dataset import MultiModalDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
import seaborn as sns
from utils.utils import plot_confusion_matrix

def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for fundus_imgs, oct_imgs, roi_masks, labels in test_loader:
            fundus_imgs = fundus_imgs.to(device)
            oct_imgs = oct_imgs.to(device)
            roi_masks = roi_masks.to(device)
            labels = labels.to(device)
            
            # Forward pass
            fusion_logits, _, _ = model(fundus_imgs, oct_imgs, roi_masks)
            probs = torch.softmax(fusion_logits, dim=1)
            
            # Get predictions
            _, preds = torch.max(fusion_logits, 1)
            
            # Store predictions and labels
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_predictions), np.array(all_probs)

def calculate_metrics(labels, predictions, probabilities, num_classes):
    # Calculate basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    # Calculate AUC for each class (one-vs-rest)
    labels_bin = label_binarize(labels, classes=range(num_classes))
    auc_scores = []
    
    for i in range(num_classes):
        try:
            auc = roc_auc_score(labels_bin[:, i], probabilities[:, i])
            auc_scores.append(auc)
        except ValueError:
            auc_scores.append(float('nan'))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_scores': auc_scores
    }

def plot_auc_curves(labels, probabilities, num_classes, class_names):
    plt.figure(figsize=(10, 8))
    labels_bin = label_binarize(labels, classes=range(num_classes))
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define parameters
    num_classes = 6  # Update this based on your dataset
    image_size = 300  # Update if using different size
    batch_size = 32
    
    # Class names for the dataset
    class_names = ['acute CSR', 'chronic CSR', 'ci-DME', 'geographic_AMD', 'Healthy', 'neovascular_AMD']  # Update with your class names
    
    # Load test dataset
    test_dataset = MultiModalDataset(
        csv_file='data/macula_test.csv',  # Update path if needed
        image_size=image_size,
        is_training=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    model = MSAN(num_classes=num_classes)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model = model.to(device)
    
    # Evaluate model
    print("Evaluating model...")
    labels, predictions, probabilities = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    metrics = calculate_metrics(labels, predictions, probabilities, num_classes)
    
    # Print results
    print("\nTest Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("\nAUC Scores per class:")
    for i, auc in enumerate(metrics['auc_scores']):
        print(f"{class_names[i]}: {auc:.4f}")
    
    # Plot ROC curves
    plot_auc_curves(labels, probabilities, num_classes, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        labels,
        predictions,
        class_names,
        title='Confusion Matrix',
        save_path='confusion_matrix.png'
    )

if __name__ == '__main__':
    main()