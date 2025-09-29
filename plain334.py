import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
from torchinfo import summary
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class PlainBlock(nn.Module):
    """
    Plain Block without residual connection.
    This is equivalent to a ResNet BasicBlock but without the skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PlainBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        # NO RESIDUAL CONNECTION HERE (this is the key difference from ResNet)
        out = F.relu(out)

        return out

class ResidualBlock(nn.Module):
    """
    Residual Block with skip connection.
    This is the key difference from PlainBlock - it includes residual connection.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        # RESIDUAL CONNECTION - This is the key difference from Plain network
        out += identity  # Add the skip connection
        out = F.relu(out)

        return out

class Plain34(nn.Module):
    """
    Plain-34 Network: ResNet-34 architecture without residual connections.
    """

    def __init__(self, num_classes=5):
        super(Plain34, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_stage(PlainBlock, 64, 64, 3, stride=1)
        self.stage2 = self._make_stage(PlainBlock, 64, 128, 4, stride=2)
        self.stage3 = self._make_stage(PlainBlock, 128, 256, 6, stride=2)
        self.stage4 = self._make_stage(PlainBlock, 256, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_stage(self, block, in_channels, out_channels, num_blocks, stride):
        downsample = None

        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))

        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class ResNet34(nn.Module):
    """
    ResNet-34 Network: ResNet-34 architecture with residual connections.
    """

    def __init__(self, num_classes=5):
        super(ResNet34, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_stage(ResidualBlock, 64, 64, 3, stride=1)
        self.stage2 = self._make_stage(ResidualBlock, 64, 128, 4, stride=2)
        self.stage3 = self._make_stage(ResidualBlock, 128, 256, 6, stride=2)
        self.stage4 = self._make_stage(ResidualBlock, 256, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_stage(self, block, in_channels, out_channels, num_blocks, stride):
        downsample = None

        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))

        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class IndonesianFoodDataset(Dataset):
    """
    Custom Dataset for Indonesian Food Classification.
    """
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.classes = sorted(self.data['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')

        label = self.data.iloc[idx]['label']
        label_idx = self.class_to_idx[label]

        if self.transform:
            image = self.transform(image)

        return image, label_idx

def get_data_transforms():
    """
    Get data transforms for training and validation.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def create_data_loaders(dataset_dir, batch_size=32, train_split=0.8):
    """
    Create train and validation data loaders.
    """
    train_csv = os.path.join(dataset_dir, 'train.csv')
    train_img_dir = os.path.join(dataset_dir, 'train')

    train_transform, val_transform = get_data_transforms()

    full_dataset = IndonesianFoodDataset(train_csv, train_img_dir, train_transform)

    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=2)

    return train_loader, val_loader, full_dataset.classes

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=10, device='cuda', model_name='Model'):
    """
    Training function for the model.
    """
    model.to(device)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    print(f"\nTraining {model_name}...")
    print("="*60)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item()
                val_running_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = val_running_corrects.double() / len(val_loader.dataset)

        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc.item())
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc.item())

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')
        print('-' * 40)

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

def evaluate_model(model, data_loader, device, class_names):
    """
    Evaluate model and return predictions and true labels for confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path=None):
    """
    Plot confusion matrix with proper formatting.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {title}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()

    return cm

def plot_comparison(plain_history, resnet_history, save_path=None):
    """
    Plot training and validation metrics comparison between Plain-34 and ResNet-34.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    epochs = range(1, len(plain_history['train_losses']) + 1)

    # Training Loss
    ax1.plot(epochs, plain_history['train_losses'], 'r-', label='Plain-34', linewidth=2)
    ax1.plot(epochs, resnet_history['train_losses'], 'b-', label='ResNet-34', linewidth=2)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Validation Loss
    ax2.plot(epochs, plain_history['val_losses'], 'r-', label='Plain-34', linewidth=2)
    ax2.plot(epochs, resnet_history['val_losses'], 'b-', label='ResNet-34', linewidth=2)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Training Accuracy
    ax3.plot(epochs, plain_history['train_accuracies'], 'r-', label='Plain-34', linewidth=2)
    ax3.plot(epochs, resnet_history['train_accuracies'], 'b-', label='ResNet-34', linewidth=2)
    ax3.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Validation Accuracy
    ax4.plot(epochs, plain_history['val_accuracies'], 'r-', label='Plain-34', linewidth=2)
    ax4.plot(epochs, resnet_history['val_accuracies'], 'b-', label='ResNet-34', linewidth=2)
    ax4.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")

    plt.show()

def print_comparison_table(plain_history, resnet_history):
    """
    Print a comparison table of final results.
    """
    print("\n" + "="*80)
    print("FINAL PERFORMANCE COMPARISON: PLAIN-34 vs RESNET-34")
    print("="*80)

    # Final metrics
    plain_final = {
        'train_acc': plain_history['train_accuracies'][-1],
        'val_acc': plain_history['val_accuracies'][-1],
        'train_loss': plain_history['train_losses'][-1],
        'val_loss': plain_history['val_losses'][-1]
    }

    resnet_final = {
        'train_acc': resnet_history['train_accuracies'][-1],
        'val_acc': resnet_history['val_accuracies'][-1],
        'train_loss': resnet_history['train_losses'][-1],
        'val_loss': resnet_history['val_losses'][-1]
    }

    print(f"{'Metric':<20} {'Plain-34':<15} {'ResNet-34':<15} {'Improvement':<15}")
    print("-" * 65)
    print(f"{'Train Accuracy':<20} {plain_final['train_acc']:<15.4f} {resnet_final['train_acc']:<15.4f} {resnet_final['train_acc'] - plain_final['train_acc']:<15.4f}")
    print(f"{'Val Accuracy':<20} {plain_final['val_acc']:<15.4f} {resnet_final['val_acc']:<15.4f} {resnet_final['val_acc'] - plain_final['val_acc']:<15.4f}")
    print(f"{'Train Loss':<20} {plain_final['train_loss']:<15.4f} {resnet_final['train_loss']:<15.4f} {resnet_final['train_loss'] - plain_final['train_loss']:<15.4f}")
    print(f"{'Val Loss':<20} {plain_final['val_loss']:<15.4f} {resnet_final['val_loss']:<15.4f} {resnet_final['val_loss'] - plain_final['val_loss']:<15.4f}")
    print("="*80)

def count_parameters(model):
    """Count the number of parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def main_comparison():
    """
    Main function to run Plain-34 vs ResNet-34 comparison experiment.
    """
    print("PLAIN-34 vs RESNET-34 COMPARISON EXPERIMENT")
    print("="*60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset directory
    dataset_dir = os.path.join('IF25-4041-dataset')

    # Check if dataset exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found!")
        print("Please make sure the dataset is in the correct location.")
        return

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, class_names = create_data_loaders(
        dataset_dir, batch_size=32, train_split=0.8
    )

    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Training configuration
    num_epochs = 10
    learning_rate = 0.001
    weight_decay = 1e-4

    print(f"\nTraining Configuration:")
    print(f"- Epochs: {num_epochs}")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Weight Decay: {weight_decay}")
    print(f"- Batch Size: 32")
    print(f"- Optimizer: Adam")

    # Create models
    print("\nCreating models...")
    plain_model = Plain34(num_classes=len(class_names))
    resnet_model = ResNet34(num_classes=len(class_names))

    # Count parameters
    plain_params, plain_trainable = count_parameters(plain_model)
    resnet_params, resnet_trainable = count_parameters(resnet_model)

    print(f"\nModel Parameters:")
    print(f"Plain-34: {plain_params:,} total, {plain_trainable:,} trainable")
    print(f"ResNet-34: {resnet_params:,} total, {resnet_trainable:,} trainable")

    # Define loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    plain_optimizer = optim.Adam(plain_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    resnet_optimizer = optim.Adam(resnet_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train Plain-34
    print("\n" + "="*60)
    print("PHASE 1: TRAINING PLAIN-34 (WITHOUT RESIDUAL CONNECTIONS)")
    print("="*60)

    start_time = time.time()
    plain_history = train_model(
        model=plain_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=plain_optimizer,
        num_epochs=num_epochs,
        device=device,
        model_name='Plain-34'
    )
    plain_train_time = time.time() - start_time

    # Train ResNet-34
    print("\n" + "="*60)
    print("PHASE 2: TRAINING RESNET-34 (WITH RESIDUAL CONNECTIONS)")
    print("="*60)

    start_time = time.time()
    resnet_history = train_model(
        model=resnet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=resnet_optimizer,
        num_epochs=num_epochs,
        device=device,
        model_name='ResNet-34'
    )
    resnet_train_time = time.time() - start_time

    # Print comparison results
    print_comparison_table(plain_history, resnet_history)

    print(f"\nTraining Time:")
    print(f"Plain-34: {plain_train_time:.2f} seconds")
    print(f"ResNet-34: {resnet_train_time:.2f} seconds")

    # Evaluate models for confusion matrix
    print("\n" + "="*60)
    print("EVALUATING MODELS FOR CONFUSION MATRIX")
    print("="*60)

    # Get predictions for both models
    plain_preds, plain_true = evaluate_model(plain_model, val_loader, device, class_names)
    resnet_preds, resnet_true = evaluate_model(resnet_model, val_loader, device, class_names)

    # Plot confusion matrices
    print("\nGenerating Confusion Matrices...")
    plain_cm = plot_confusion_matrix(plain_true, plain_preds, class_names,
                                   "Plain-34", "plain34_confusion_matrix.png")
    resnet_cm = plot_confusion_matrix(resnet_true, resnet_preds, class_names,
                                    "ResNet-34", "resnet34_confusion_matrix.png")

    # Print classification reports
    print("\n" + "="*60)
    print("PLAIN-34 CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(plain_true, plain_preds, target_names=class_names))

    print("\n" + "="*60)
    print("RESNET-34 CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(resnet_true, resnet_preds, target_names=class_names))

    # Plot comparison
    plot_comparison(plain_history, resnet_history, 'plain_vs_resnet_comparison.png')

    # Save results
    results_dir = 'ResNet'
    os.makedirs(results_dir, exist_ok=True)

    # Save Plain-34 model and results
    plain_save_path = os.path.join(results_dir, 'plain34_trained.pth')
    torch.save({
        'model_state_dict': plain_model.state_dict(),
        'optimizer_state_dict': plain_optimizer.state_dict(),
        'history': plain_history,
        'class_names': class_names,
        'training_time': plain_train_time
    }, plain_save_path)

    # Save ResNet-34 model and results
    resnet_save_path = os.path.join(results_dir, 'resnet34_trained.pth')
    torch.save({
        'model_state_dict': resnet_model.state_dict(),
        'optimizer_state_dict': resnet_optimizer.state_dict(),
        'history': resnet_history,
        'class_names': class_names,
        'training_time': resnet_train_time
    }, resnet_save_path)

    # Save comparison results
    comparison_results = {
        'plain_history': plain_history,
        'resnet_history': resnet_history,
        'plain_train_time': plain_train_time,
        'resnet_train_time': resnet_train_time,
        'plain_params': plain_params,
        'resnet_params': resnet_params,
        'plain_confusion_matrix': plain_cm,
        'resnet_confusion_matrix': resnet_cm,
        'plain_predictions': plain_preds,
        'resnet_predictions': resnet_preds,
        'true_labels': plain_true,  # Both models use same validation set
        'class_names': class_names,
        'configuration': {
            'epochs': num_epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'batch_size': 32,
            'optimizer': 'Adam'
        }
    }

    comparison_save_path = os.path.join(results_dir, 'comparison_results.pth')
    torch.save(comparison_results, comparison_save_path)

    print(f"\nModels and results saved:")
    print(f"- Plain-34: {plain_save_path}")
    print(f"- ResNet-34: {resnet_save_path}")
    print(f"- Comparison: {comparison_save_path}")

    # Analysis summary
    val_acc_improvement = resnet_history['val_accuracies'][-1] - plain_history['val_accuracies'][-1]
    val_loss_improvement = plain_history['val_losses'][-1] - resnet_history['val_losses'][-1]

    print(f"\n" + "="*60)
    print("EXPERIMENT ANALYSIS SUMMARY")
    print("="*60)
    print(f"Validation Accuracy Improvement: {val_acc_improvement:.4f} ({val_acc_improvement*100:.2f}%)")
    print(f"Validation Loss Improvement: {val_loss_improvement:.4f}")

    if val_acc_improvement > 0:
        print("✓ ResNet-34 shows better performance than Plain-34")
        print("✓ Residual connections successfully help with training deeper networks")
    else:
        print("⚠ Plain-34 performed unexpectedly better - consider checking implementation")

    print("="*60)

    return plain_model, resnet_model, plain_history, resnet_history

if __name__ == "__main__":
    # Run the comparison experiment
    plain_model, resnet_model, plain_history, resnet_history = main_comparison()