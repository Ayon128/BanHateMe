import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
import random

import config
from models import create_text_encoder, create_image_encoder, ProjectionLayer
from fusion import create_fusion_module
from dataset import create_dataloaders
from utils import (
    calculate_metrics,
    save_checkpoint,
    load_checkpoint,
    plot_training_history,
    save_confusion_matrices
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class HierarchicalMultimodalModel(nn.Module):
    """
    Hierarchical Multi-modal Model for Hateful Meme Classification

    Architecture:
        1. Text Encoder (BanglaBERT or XLM-RoBERTa)
        2. Image Encoder (ViT or Swin Transformer)
        3. Fusion Module (Summation, Concatenation, or Co-Attention)
        4. Three Classification Heads:
           - Binary: Hate vs Non-Hate (2 classes)
           - Category: Hate categories (5 classes)
           - Target: Target groups (4 classes)
    """

    def __init__(self, text_encoder_name, image_encoder_name, fusion_type):
        super(HierarchicalMultimodalModel, self).__init__()

        # Create encoders
        self.text_encoder = create_text_encoder(text_encoder_name)
        self.image_encoder = create_image_encoder(image_encoder_name)

        # Get hidden sizes
        text_hidden_size = self.text_encoder.hidden_size
        image_hidden_size = self.image_encoder.hidden_size

        # Project to common hidden size if needed
        self.text_projection = ProjectionLayer(text_hidden_size, config.HIDDEN_SIZE)
        self.image_projection = ProjectionLayer(image_hidden_size, config.HIDDEN_SIZE)

        # Create fusion module
        self.fusion = create_fusion_module(fusion_type, config.HIDDEN_SIZE)
        self.fusion_type = fusion_type

        # Classification heads
        self.binary_classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE // 2, config.NUM_BINARY_CLASSES)
        )

        self.category_classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE // 2, config.NUM_CATEGORY_CLASSES)
        )

        self.target_classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE // 2, config.NUM_TARGET_CLASSES)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            pixel_values: [batch_size, 3, height, width]

        Returns:
            binary_logits: [batch_size, 2]
            category_logits: [batch_size, 5]
            target_logits: [batch_size, 4]
        """
        # Encode text and image
        text_cls, text_all = self.text_encoder(input_ids, attention_mask)
        image_cls, image_all = self.image_encoder(pixel_values)

        # Project to common size
        text_cls = self.text_projection(text_cls)
        text_all = self.text_projection(text_all)
        image_cls = self.image_projection(image_cls)
        image_all = self.image_projection(image_all)

        # Fuse modalities
        fused = self.fusion(text_cls, image_cls, text_all, image_all)

        # Classify at three levels
        binary_logits = self.binary_classifier(fused)
        category_logits = self.category_classifier(fused)
        target_logits = self.target_classifier(fused)

        return binary_logits, category_logits, target_logits


def hierarchical_loss(binary_logits, category_logits, target_logits,
                     binary_labels, category_labels, target_labels,
                     alpha, beta):
    """
    Hierarchical loss function from paper:
    L_total = L_binary + α * L_category + β * L_target

    Args:
        binary_logits, category_logits, target_logits: Model outputs
        binary_labels, category_labels, target_labels: Ground truth labels
        alpha: Weight for category loss
        beta: Weight for target loss

    Returns:
        total_loss, binary_loss, category_loss, target_loss
    """
    criterion = nn.CrossEntropyLoss()

    binary_loss = criterion(binary_logits, binary_labels)
    category_loss = criterion(category_logits, category_labels)
    target_loss = criterion(target_logits, target_labels)

    total_loss = binary_loss + alpha * category_loss + beta * target_loss

    return total_loss, binary_loss, category_loss, target_loss


def train_epoch(model, dataloader, optimizer, scaler, device, alpha, beta, use_amp=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_binary_loss = 0
    total_category_loss = 0
    total_target_loss = 0

    all_binary_preds = []
    all_binary_labels = []

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        binary_labels = batch['binary_label'].to(device)
        category_labels = batch['category_label'].to(device)
        target_labels = batch['target_label'].to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision if GPU
        if use_amp:
            with torch.amp.autocast("cuda"):
                binary_logits, category_logits, target_logits = model(
                    input_ids, attention_mask, pixel_values
                )
                loss, binary_loss, category_loss, target_loss = hierarchical_loss(
                    binary_logits, category_logits, target_logits,
                    binary_labels, category_labels, target_labels,
                    alpha, beta
                )

            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training 
            binary_logits, category_logits, target_logits = model(
                input_ids, attention_mask, pixel_values
            )
            loss, binary_loss, category_loss, target_loss = hierarchical_loss(
                binary_logits, category_logits, target_logits,
                binary_labels, category_labels, target_labels,
                alpha, beta
            )

            # Backward pass
            loss.backward()
            optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_binary_loss += binary_loss.item()
        total_category_loss += category_loss.item()
        total_target_loss += target_loss.item()

        # Get predictions
        binary_preds = torch.argmax(binary_logits, dim=1)
        all_binary_preds.extend(binary_preds.cpu().numpy())
        all_binary_labels.extend(binary_labels.cpu().numpy())

        # Log progress
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            print(f"  Batch [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"(Binary: {binary_loss.item():.4f}, "
                  f"Cat: {category_loss.item():.4f}, "
                  f"Target: {target_loss.item():.4f})")

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    avg_binary_loss = total_binary_loss / len(dataloader)
    avg_category_loss = total_category_loss / len(dataloader)
    avg_target_loss = total_target_loss / len(dataloader)

    binary_acc = np.mean(np.array(all_binary_preds) == np.array(all_binary_labels))

    return {
        'loss': avg_loss,
        'binary_loss': avg_binary_loss,
        'category_loss': avg_category_loss,
        'target_loss': avg_target_loss,
        'binary_acc': binary_acc
    }


@torch.no_grad()
def evaluate(model, dataloader, device, alpha, beta, use_amp=False):
    """Evaluate the model"""
    model.eval()
    total_loss = 0

    all_binary_preds = []
    all_binary_labels = []
    all_category_preds = []
    all_category_labels = []
    all_target_preds = []
    all_target_labels = []

    for batch in dataloader:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        binary_labels = batch['binary_label'].to(device)
        category_labels = batch['category_label'].to(device)
        target_labels = batch['target_label'].to(device)

        # Forward pass
        if use_amp:
            with torch.amp.autocast("cuda"):
                binary_logits, category_logits, target_logits = model(
                    input_ids, attention_mask, pixel_values
                )
                loss, _, _, _ = hierarchical_loss(
                    binary_logits, category_logits, target_logits,
                    binary_labels, category_labels, target_labels,
                    alpha, beta
                )
        else:
            binary_logits, category_logits, target_logits = model(
                input_ids, attention_mask, pixel_values
            )
            loss, _, _, _ = hierarchical_loss(
                binary_logits, category_logits, target_logits,
                binary_labels, category_labels, target_labels,
                alpha, beta
            )

        total_loss += loss.item()

        # Get predictions
        binary_preds = torch.argmax(binary_logits, dim=1)
        category_preds = torch.argmax(category_logits, dim=1)
        target_preds = torch.argmax(target_logits, dim=1)

        all_binary_preds.extend(binary_preds.cpu().numpy())
        all_binary_labels.extend(binary_labels.cpu().numpy())
        all_category_preds.extend(category_preds.cpu().numpy())
        all_category_labels.extend(category_labels.cpu().numpy())
        all_target_preds.extend(target_preds.cpu().numpy())
        all_target_labels.extend(target_labels.cpu().numpy())

    # Calculate metrics for all levels
    metrics = calculate_metrics(
        all_binary_preds, all_binary_labels,
        all_category_preds, all_category_labels,
        all_target_preds, all_target_labels
    )

    metrics['loss'] = total_loss / len(dataloader)

    return metrics, (all_binary_preds, all_binary_labels,
                     all_category_preds, all_category_labels,
                     all_target_preds, all_target_labels)


def train_model():
    """Main training function"""
    print("="*70)
    print("BLP BANGLA HATEFUL MEME CLASSIFICATION")
    print("="*70)
    print(f"Text Encoder: {config.TEXT_ENCODER}")
    print(f"Image Encoder: {config.IMAGE_ENCODER}")
    print(f"Fusion Type: {config.FUSION_TYPE}")
    print(f"Alpha: {config.ALPHA}, Beta: {config.BETA}")
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.EPOCHS}")
    print("="*70)

    # Set seed
    set_seed(config.RANDOM_SEED)

    # Create model
    print("\nInitializing model...")
    model = HierarchicalMultimodalModel(
        config.TEXT_ENCODER,
        config.IMAGE_ENCODER,
        config.FUSION_TYPE
    )
    model = model.to(config.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        text_tokenizer=model.text_encoder.get_tokenizer(),
        image_processor=model.image_encoder.get_processor(),
        batch_size=config.BATCH_SIZE,
        num_workers=0  
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Scaler for mixed precision (only on GPU)
    scaler = GradScaler('cuda') if config.USE_AMP else None

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    best_val_f1 = 0
    patience_counter = 0

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        print(f"\nEpoch [{epoch + 1}/{config.EPOCHS}]")
        print("-" * 70)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler,
            config.DEVICE, config.ALPHA, config.BETA,
            use_amp=config.USE_AMP
        )

        # Validate
        print("\nValidating...")
        val_metrics, _ = evaluate(
            model, val_loader,
            config.DEVICE, config.ALPHA, config.BETA,
            use_amp=config.USE_AMP
        )

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['binary_acc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['binary']['accuracy'])
        history['val_f1'].append(val_metrics['binary']['f1'])

        epoch_time = time.time() - start_time

        # Print epoch summary
        print(f"\nEpoch Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['binary_acc']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['binary']['accuracy']:.4f}, "
              f"F1: {val_metrics['binary']['f1']:.4f}")
        print(f"  Time: {epoch_time:.2f}s")

        # Save best model
        if val_metrics['binary']['f1'] > best_val_f1:
            best_val_f1 = val_metrics['binary']['f1']
            patience_counter = 0
            if config.SAVE_BEST_MODEL:
                save_checkpoint(model, optimizer, epoch, val_metrics, config.OUTPUT_DIR, 'best')
                print(f"  ✓ New best model saved! F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Save final model
    if config.SAVE_LAST_MODEL:
        save_checkpoint(model, optimizer, epoch, val_metrics, config.OUTPUT_DIR, 'last')

    # Save training history
    plot_training_history(history, config.OUTPUT_DIR)

    # Final evaluation on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)

    # Load best model
    checkpoint_path = os.path.join(config.OUTPUT_DIR, 'model_best.pth')
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, optimizer, checkpoint_path)

    test_metrics, test_predictions = evaluate(
        model, test_loader,
        config.DEVICE, config.ALPHA, config.BETA,
        use_amp=config.USE_AMP
    )

    # Print test results
    print("\nTest Results:")
    print(f"\nBinary Classification (Hate vs Non-Hate):")
    print(f"  Accuracy: {test_metrics['binary']['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['binary']['precision']:.4f}")
    print(f"  Recall: {test_metrics['binary']['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['binary']['f1']:.4f}")

    print(f"\nCategory Classification:")
    print(f"  Accuracy: {test_metrics['category']['accuracy']:.4f}")
    print(f"  F1 Score: {test_metrics['category']['f1']:.4f}")

    print(f"\nTarget Group Classification:")
    print(f"  Accuracy: {test_metrics['target']['accuracy']:.4f}")
    print(f"  F1 Score: {test_metrics['target']['f1']:.4f}")

    # Save results
    if config.SAVE_RESULTS:
        results = {
            'config': {
                'text_encoder': config.TEXT_ENCODER,
                'image_encoder': config.IMAGE_ENCODER,
                'fusion_type': config.FUSION_TYPE,
                'alpha': config.ALPHA,
                'beta': config.BETA,
            },
            'test_metrics': test_metrics,
            'training_history': history
        }

        results_path = os.path.join(config.OUTPUT_DIR, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")

    # Save confusion matrices
    if config.SAVE_CONFUSION_MATRIX:
        save_confusion_matrices(test_predictions, config.OUTPUT_DIR)

    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    train_model()
