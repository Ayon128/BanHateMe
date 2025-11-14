import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import config


def calculate_metrics(binary_preds, binary_labels,
                     category_preds, category_labels,
                     target_preds, target_labels):
    """
    Calculate comprehensive metrics for all three classification levels

    Args:
        binary_preds, binary_labels: Predictions and labels for binary classification
        category_preds, category_labels: Predictions and labels for category classification
        target_preds, target_labels: Predictions and labels for target classification

    Returns:
        dict with metrics for each level
    """
    metrics = {}

    # Binary classification metrics
    metrics['binary'] = {
        'accuracy': accuracy_score(binary_labels, binary_preds),
        'precision': precision_score(binary_labels, binary_preds, average='weighted', zero_division=0),
        'recall': recall_score(binary_labels, binary_preds, average='weighted', zero_division=0),
        'f1': f1_score(binary_labels, binary_preds, average='weighted', zero_division=0)
    }

    # Category classification metrics
    metrics['category'] = {
        'accuracy': accuracy_score(category_labels, category_preds),
        'precision': precision_score(category_labels, category_preds, average='weighted', zero_division=0),
        'recall': recall_score(category_labels, category_preds, average='weighted', zero_division=0),
        'f1': f1_score(category_labels, category_preds, average='weighted', zero_division=0)
    }

    # Target group classification metrics
    metrics['target'] = {
        'accuracy': accuracy_score(target_labels, target_preds),
        'precision': precision_score(target_labels, target_preds, average='weighted', zero_division=0),
        'recall': recall_score(target_labels, target_preds, average='weighted', zero_division=0),
        'f1': f1_score(target_labels, target_preds, average='weighted', zero_division=0)
    }

    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, output_dir, checkpoint_name='best'):
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Validation metrics
        output_dir: Directory to save checkpoint
        checkpoint_name: Name for checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    checkpoint_path = os.path.join(output_dir, f'model_{checkpoint_name}.pth')
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint file

    Returns:
        epoch, metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"✓ Checkpoint loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")

    return checkpoint['epoch'], checkpoint['metrics']


def plot_training_history(history, output_dir):
    """
    Plot and save training history

    Args:
        history: Dictionary with training history
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].plot(history['val_f1'], label='Val F1', marker='^')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Training and Validation Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Training history plot saved to {plot_path}")


def save_confusion_matrices(predictions, output_dir):
    """
    Save confusion matrices for all three classification levels

    Args:
        predictions: Tuple of (binary_preds, binary_labels, category_preds, category_labels,
                               target_preds, target_labels)
        output_dir: Directory to save plots
    """
    binary_preds, binary_labels, category_preds, category_labels, target_preds, target_labels = predictions

    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)

    # Binary confusion matrix
    cm_binary = confusion_matrix(binary_labels, binary_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Hate', 'Hate'],
                yticklabels=['Non-Hate', 'Hate'])
    plt.title('Binary Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(cm_dir, 'binary_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Category confusion matrix
    cm_category = confusion_matrix(category_labels, category_preds)
    plt.figure(figsize=(10, 8))
    category_names = ['Abusive', 'Political', 'Gender', 'Personal', 'Religious']
    sns.heatmap(cm_category, annot=True, fmt='d', cmap='Blues',
                xticklabels=category_names,
                yticklabels=category_names)
    plt.title('Category Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(cm_dir, 'category_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Target confusion matrix
    cm_target = confusion_matrix(target_labels, target_preds)
    plt.figure(figsize=(10, 8))
    target_names = ['Community', 'Individual', 'Organization', 'Society']
    sns.heatmap(cm_target, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title('Target Group Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(cm_dir, 'target_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Confusion matrices saved to {cm_dir}")

    # Save classification reports
    report_path = os.path.join(cm_dir, 'classification_reports.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BINARY CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n")
        f.write(classification_report(binary_labels, binary_preds,
                                     target_names=['Non-Hate', 'Hate']))

        f.write("\n" + "="*70 + "\n")
        f.write("CATEGORY CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n")
        f.write(classification_report(category_labels, category_preds,
                                     target_names=category_names))

        f.write("\n" + "="*70 + "\n")
        f.write("TARGET GROUP CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n")
        f.write(classification_report(target_labels, target_preds,
                                     target_names=target_names))

    print(f"✓ Classification reports saved to {report_path}")
