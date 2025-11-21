import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import StepLR
from utils.FocalLoss import FocalLoss
import matplotlib.pyplot as plt  # <--- ADDED IMPORT

# Import our custom classes
from data.dataset import MultiModalDataset
from model.msan import MSAN

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_fusion = 0
    total = 0

    for fundus, oct, roi, labels in tqdm(dataloader, desc="Training"):
        fundus, oct, roi, labels = fundus.to(device), oct.to(device), roi.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        fusion_logits, fundus_logits, oct_logits = model(fundus, oct, roi)
        
        # Calculate loss
        loss_fusion = criterion(fusion_logits, labels)
        loss_fundus = criterion(fundus_logits, labels)
        loss_oct = criterion(oct_logits, labels)
        
        # Combined loss from the paper
        loss = 0.5 * loss_fundus + 0.5 * loss_oct + loss_fusion

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(fusion_logits.data, 1)
        total += labels.size(0)
        correct_fusion += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct_fusion / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_fusion = 0
    total = 0

    with torch.no_grad():
        for fundus, oct, roi, labels in tqdm(dataloader, desc="Validating"):
            fundus, oct, roi, labels = fundus.to(device), oct.to(device), roi.to(device), labels.to(device)

            fusion_logits, fundus_logits, oct_logits = model(fundus, oct, roi)
            
            loss_fusion = criterion(fusion_logits, labels)
            loss_fundus = criterion(fundus_logits, labels)
            loss_oct = criterion(oct_logits, labels)
            loss = 0.5 * loss_fundus + 0.5 * loss_oct + loss_fusion
            
            running_loss += loss.item()
            _, predicted = torch.max(fusion_logits.data, 1)
            total += labels.size(0)
            correct_fusion += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct_fusion / total
    return epoch_loss, epoch_acc

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets and Dataloaders
    train_dataset = MultiModalDataset(csv_file=args.train_csv, image_size=args.image_size, is_training=True)
    val_dataset = MultiModalDataset(csv_file=args.val_csv, image_size=args.image_size, is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model, Loss, Optimizer, and Scheduler
    model = MSAN(num_classes=args.num_classes).to(device)
    print(f"Model created with {args.num_classes} classes.")
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # --- ADDED: Initialize history lists to store metrics ---
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # --- Training loop ---
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # --- ADDED: Append metrics for this epoch to the history ---
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_path) # Use model_path from args
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")

    print("\n--- Training Complete ---")
    
    # --- ADDED: Plotting the convergence graph ---
    print("Generating convergence graph...")
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid') # Fallback for older matplotlib versions

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Plot Loss
    ax1.plot(history['train_loss'], label='Training Loss', color='dodgerblue', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', color='darkorange', linestyle='--', linewidth=2)
    ax1.set_title('Model Loss Convergence', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='upper right')

    # Plot Accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy', color='dodgerblue', linewidth=2)
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='darkorange', linestyle='--', linewidth=2)
    ax2.set_title('Model Accuracy Convergence', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(loc='lower right')
    
    fig.suptitle('Training & Validation Metrics', fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    graph_filename = 'convergence_graph.png'
    plt.savefig(graph_filename)
    print(f"Convergence graph saved as '{graph_filename}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MSAN on Retinal Images')
    parser.add_argument('--train_csv', type=str, default='data/macula_train.csv', help='Path to the training CSV file')
    parser.add_argument('--val_csv', type=str, default='data/macula_val.csv', help='Path to the validation CSV file')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to save the best model')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes in the dataset')
    parser.add_argument('--image_size', type=int, default=300, help='Size to resize images to')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    
    args = parser.parse_args()
    main(args)