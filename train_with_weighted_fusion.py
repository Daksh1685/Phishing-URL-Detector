"""
WEIGHTED FUSION TRAINING - Start Here!

This script:
1. Loads individual embeddings (w2v, fasttext, glove/bert, roberta, gpt2)
2. Applies weighted fusion (learns 6 importance weights)
3. Trains CNN on fused embeddings
4. Saves best model

Expected accuracy improvement: 98.35% → 98.5-99.0% (+0.2-0.7%)
Time to train: ~2-3 hours on GTX 1650
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from models.cnn_model import CNN1D
from embeddings.fusion_advanced import FusionLayer
from utils.device import get_device
from utils.logger import setup_logger

# Setup
device = get_device()
logger = setup_logger("WeightedFusion", "logs/weighted_fusion.log")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'embedding_dir': 'data/embeddings',
    'batch_size': 128,
    'epochs': 20,
    'lr': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 5,
    'checkpoint_dir': 'training/checkpoints/weighted_fusion',
    'fusion_mode': 'weighted',  # 6 learnable parameters
    'output_dim': 2504,  # Keep same as original concatenation
}

# ============================================================================
# LOAD EMBEDDINGS
# ============================================================================

def load_embeddings():
    """Load all individual embeddings from .npy files"""
    
    logger.info("=" * 80)
    logger.info("LOADING EMBEDDINGS")
    logger.info("=" * 80)
    
    emb_dir = Path(CONFIG['embedding_dir'])
    
    # Load training embeddings
    logger.info("\nTraining embeddings:")
    train_embeddings = []
    embedding_names = ['w2v', 'fasttext', 'glove', 'bert', 'roberta', 'gpt2']
    embedding_dims = []
    loaded_names = []  # Track which embeddings were successfully loaded
    
    for name in embedding_names:
        # Try to find the file
        pattern = f"train_{name}_vectors.npy"
        files = list(emb_dir.glob(pattern))
        
        if files:
            path = files[0]
            emb = np.load(path)
            train_embeddings.append(torch.tensor(emb).float().to(device))
            dim = emb.shape[1]
            embedding_dims.append(dim)
            loaded_names.append(name)  # Track successfully loaded name
            logger.info(f"  [OK] {name:12} : {emb.shape} ({dim:4}-dim)")
        else:
            logger.warning(f"  [XX] {name:12} : NOT FOUND")
    
    # Load test embeddings
    logger.info("\nTest embeddings:")
    test_embeddings = []
    
    for name in loaded_names:  # Use the names that were actually loaded for training
        pattern = f"test_{name}_vectors.npy"
        files = list(emb_dir.glob(pattern))
        
        if files:
            path = files[0]
            emb = np.load(path)
            test_embeddings.append(torch.tensor(emb).float().to(device))
            logger.info(f"  [OK] {name:12} : {emb.shape}")
        else:
            logger.warning(f"  [XX] {name:12} : NOT FOUND")
    
    # Load labels
    logger.info("\nLabels:")
    train_labels = torch.tensor(np.load(emb_dir / 'train_labels.npy')).long().to(device)
    test_labels = torch.tensor(np.load(emb_dir / 'test_labels.npy')).long().to(device)
    logger.info(f"  ✓ train_labels: {train_labels.shape}")
    logger.info(f"  ✓ test_labels : {test_labels.shape}")
    
    logger.info(f"\n[Summary] Loaded {len(train_embeddings)} embeddings")
    logger.info(f"          Embedding dims: {embedding_dims}")
    logger.info(f"          Training samples: {train_labels.shape[0]}")
    logger.info(f"          Test samples: {test_labels.shape[0]}")
    
    return train_embeddings, test_embeddings, train_labels, test_labels, embedding_dims

# ============================================================================
# CREATE FUSION LAYER
# ============================================================================

def create_fusion_layer(embedding_dims):
    """Create weighted fusion layer"""
    
    logger.info("\n" + "=" * 80)
    logger.info("CREATING FUSION LAYER")
    logger.info("=" * 80)
    
    fusion = FusionLayer(
        embedding_dims=embedding_dims,
        num_embeddings=len(embedding_dims),
        mode='weighted',
        output_dim=CONFIG['output_dim']
    ).to(device)
    
    num_params = sum(p.numel() for p in fusion.parameters())
    logger.info(f"\n[OK] Weighted Fusion Layer Created")
    logger.info(f"  Mode: weighted")
    logger.info(f"  Input dims: {embedding_dims}")
    logger.info(f"  Output dim: {CONFIG['output_dim']}")
    logger.info(f"  Learnable params: {num_params}")
    
    return fusion

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, fusion, train_embeddings, train_labels, optimizer, criterion, epoch, num_epochs):
    """Train one epoch"""
    
    model.train()
    fusion.train()
    
    num_batches = (len(train_labels) + CONFIG['batch_size'] - 1) // CONFIG['batch_size']
    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False)
    
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_idx in pbar:
        start = batch_idx * CONFIG['batch_size']
        end = min((batch_idx + 1) * CONFIG['batch_size'], len(train_labels))
        
        # Get batch embeddings
        batch_embeddings = [emb[start:end] for emb in train_embeddings]
        batch_labels = train_labels[start:end]
        
        # Forward pass through fusion
        fused = fusion(batch_embeddings)
        
        # Forward pass through CNN
        logits = model(fused)
        loss = criterion(logits, batch_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        train_loss += loss.item()
        preds = logits.argmax(1)
        train_correct += (preds == batch_labels).sum().item()
        train_total += batch_labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    train_loss /= num_batches
    train_acc = train_correct / train_total
    
    return train_loss, train_acc

# ============================================================================
# VALIDATION
# ============================================================================

def validate(model, fusion, test_embeddings, test_labels, criterion, epoch, num_epochs):
    """Validate model"""
    
    model.eval()
    fusion.eval()
    
    num_batches = (len(test_labels) + CONFIG['batch_size'] - 1) // CONFIG['batch_size']
    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{num_epochs} [Val]", leave=False)
    
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_idx in pbar:
            start = batch_idx * CONFIG['batch_size']
            end = min((batch_idx + 1) * CONFIG['batch_size'], len(test_labels))
            
            batch_embeddings = [emb[start:end] for emb in test_embeddings]
            batch_labels = test_labels[start:end]
            
            # Fuse then classify
            fused = fusion(batch_embeddings)
            logits = model(fused)
            loss = criterion(logits, batch_labels)
            
            val_loss += loss.item()
            preds = logits.argmax(1)
            val_correct += (preds == batch_labels).sum().item()
            val_total += batch_labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    val_loss /= num_batches
    val_acc = val_correct / val_total
    
    return val_loss, val_acc

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    """Main training function"""
    
    logger.info("\n" * 2)
    logger.info("=" * 80)
    logger.info("WEIGHTED FUSION TRAINING - START".center(80))
    logger.info("=" * 80)
    
    logger.info(f"\nDevice: {device}")
    logger.info(f"Config: {CONFIG}")
    
    # ======================================================================
    # 1. LOAD EMBEDDINGS
    # ======================================================================
    
    train_embeddings, test_embeddings, train_labels, test_labels, embedding_dims = load_embeddings()
    
    # ======================================================================
    # 2. CREATE MODELS
    # ======================================================================
    
    # Fusion layer
    fusion = create_fusion_layer(embedding_dims)
    
    # CNN model
    logger.info("\n" + "=" * 80)
    logger.info("CREATING CNN MODEL")
    logger.info("=" * 80)
    
    model = CNN1D(
        input_dim=CONFIG['output_dim'],
        num_classes=2,
        conv_channels=(32, 64, 128, 256),
        kernel_sizes=(5, 5, 3, 3),
        pool_sizes=(2, 2, 2, 2),
        fc_dims=(512, 256),
        dropout_rate=0.3,
        fc_dropout=0.4
    ).to(device)
    
    num_cnn_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\n[OK] CNN Model Created")
    logger.info(f"  Input dim: {CONFIG['output_dim']}")
    logger.info(f"  Output classes: 2")
    logger.info(f"  Total CNN params: {num_cnn_params:,}")
    
    # ======================================================================
    # 3. SETUP TRAINING
    # ======================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("SETUP TRAINING")
    logger.info("=" * 80)
    
    # Optimizer (for both fusion and CNN)
    all_params = list(model.parameters()) + list(fusion.parameters())
    optimizer = optim.Adam(all_params, lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    criterion = nn.CrossEntropyLoss()
    
    logger.info(f"\n[OK] Optimizer: Adam")
    logger.info(f"  Learning rate: {CONFIG['lr']}")
    logger.info(f"  Weight decay: {CONFIG['weight_decay']}")
    logger.info(f"  Total parameters: {num_cnn_params + sum(p.numel() for p in fusion.parameters()):,}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(CONFIG['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # ======================================================================
    # 4. TRAINING LOOP
    # ======================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING")
    logger.info("=" * 80)
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        
        # Train
        train_loss, train_acc = train_epoch(
            model, fusion, train_embeddings, train_labels, 
            optimizer, criterion, epoch, CONFIG['epochs']
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, fusion, test_embeddings, test_labels, 
            criterion, epoch, CONFIG['epochs']
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log
        logger.info(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        logger.info(f"  Train: Loss {train_loss:.4f}, Acc {train_acc:.4f}")
        logger.info(f"  Val:   Loss {val_loss:.4f}, Acc {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save
            checkpoint_path = checkpoint_dir / 'best_weighted_fusion_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'fusion_state_dict': fusion.state_dict(),
                'val_acc': val_acc,
                'embedding_dims': embedding_dims,
            }, checkpoint_path)
            
            logger.info(f"  [BEST] Saved model with acc {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stopping_patience']:
                logger.info(f"\n[EARLY STOPPING] No improvement for {CONFIG['early_stopping_patience']} epochs")
                break
    
    # ======================================================================
    # 5. FINAL EVALUATION
    # ======================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 80)
    
    # Load best model
    checkpoint_path = checkpoint_dir / 'best_weighted_fusion_model.pt'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    fusion.load_state_dict(checkpoint['fusion_state_dict'])
    
    # Evaluate on full test set
    model.eval()
    fusion.eval()
    
    with torch.no_grad():
        # Get predictions on all test data
        all_preds = []
        all_labels = []
        
        num_batches = (len(test_labels) + CONFIG['batch_size'] - 1) // CONFIG['batch_size']
        for batch_idx in range(num_batches):
            start = batch_idx * CONFIG['batch_size']
            end = min((batch_idx + 1) * CONFIG['batch_size'], len(test_labels))
            
            batch_embeddings = [emb[start:end] for emb in test_embeddings]
            batch_labels = test_labels[start:end]
            
            fused = fusion(batch_embeddings)
            logits = model(fused)
            preds = logits.argmax(1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    
    logger.info(f"\n[OK] Test Set Accuracy: {accuracy:.4f}")
    logger.info(f"  Baseline (concatenation): 0.9835 (98.35%)")
    logger.info(f"  Improvement: +{(accuracy - 0.9835):.4f} (+{(accuracy/0.9835 - 1)*100:.2f}%)")
    
    # Save fusion weights
    if hasattr(fusion, 'weights'):
        weights = F.softmax(fusion.weights, dim=0)
        names = ['W2V', 'FT', 'BERT', 'RoBERTa', 'GPT-2'][:len(weights)]
        logger.info(f"\nLearned Fusion Weights:")
        for name, w in zip(names, weights):
            logger.info(f"  {name:10} : {w.item():.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nModel saved to: {checkpoint_path}")
    logger.info(f"Logs saved to: logs/weighted_fusion.log")
    logger.info("\nNext: Run evaluation/evaluate_weighted_fusion.py to visualize results")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()
