import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from pathlib import Path
import time
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cnn_model import CNN1D, SimpleCNN1D, DeepCNN1D


class CNNTrainer:
    """Trainer for 1D CNN models."""
    
    def __init__(self, model, device='cuda', learning_rate=0.001, weight_decay=1e-5):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler for learning rate decay
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=20
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)
            
            pbar.set_postfix(loss=loss.item(), acc=100*correct/total)
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100 * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += y_batch.size(0)
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = 100 * correct / total
        
        return avg_loss, avg_acc
    
    def train(self, train_loader, val_loader, epochs=20, patience=5):
        """
        Train model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        best_val_acc = 0
        patience_counter = 0
        
        print(f"\n{'=' * 80}")
        print(f"TRAINING CNN1D MODEL")
        print(f"{'=' * 80}")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Epochs: {epochs}\n")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"{'-' * 80}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.6f} | Val Acc:   {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                print(f"✓ Better validation accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠ Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\n{'=' * 80}")
        print(f"TRAINING COMPLETED")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"{'=' * 80}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
    
    def evaluate(self, test_loader):
        """
        Evaluate on test set with detailed metrics.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_logits = []
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits = self.model(X_batch)
                probs = torch.softmax(logits, dim=1)
                _, predicted = logits.max(1)
                
                all_logits.append(logits.cpu())
                all_preds.append(predicted.cpu())
                all_true.append(y_batch.cpu())
        
        # Concatenate all batches
        all_logits = torch.cat(all_logits)
        all_preds = torch.cat(all_preds)
        all_true = torch.cat(all_true)
        
        # Calculate metrics
        correct = (all_preds == all_true).sum().item()
        accuracy = 100 * correct / len(all_true)
        
        # Per-class metrics
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true.numpy(), all_preds.numpy(), average='weighted'
        )
        
        # Binary ROC-AUC
        probs = torch.softmax(all_logits, dim=1)[:, 1].numpy()
        roc_auc = roc_auc_score(all_true.numpy(), probs)
        
        # Confusion matrix
        cm = confusion_matrix(all_true.numpy(), all_preds.numpy())
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': all_preds.numpy(),
            'true_labels': all_true.numpy(),
            'probabilities': probs
        }


def load_embeddings(data_dir='./data/embeddings', use_attention=True):
    """
    Load embeddings (fused or attention-weighted).
    
    Args:
        data_dir: Directory containing embedding files
        use_attention: Use attention-weighted embeddings if available
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"\nLoading embeddings from {data_dir}...")
    
    if use_attention:
        # Try to load attention-weighted embeddings first
        try:
            train_path = os.path.join(data_dir, 'train_attended_multi_head_vectors.npy')
            test_path = os.path.join(data_dir, 'test_attended_multi_head_vectors.npy')
            
            if os.path.exists(train_path) and os.path.exists(test_path):
                X_train = np.load(train_path)
                X_test = np.load(test_path)
                print(f"✓ Using attention-weighted embeddings")
            else:
                raise FileNotFoundError("Attention embeddings not found")
        except Exception as e:
            print(f"⚠ Attention embeddings not available: {e}")
            print(f"  Falling back to fused embeddings...")
            X_train = np.load(os.path.join(data_dir, 'train_fused_vectors.npy'))
            X_test = np.load(os.path.join(data_dir, 'test_fused_vectors.npy'))
    else:
        X_train = np.load(os.path.join(data_dir, 'train_fused_vectors.npy'))
        X_test = np.load(os.path.join(data_dir, 'test_fused_vectors.npy'))
    
    y_train = np.load(os.path.join(data_dir, 'train_labels.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy'))
    
    print(f"✓ X_train: {X_train.shape}")
    print(f"✓ X_test:  {X_test.shape}")
    print(f"✓ y_train: {y_train.shape}")
    print(f"✓ y_test:  {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_cnn_model(model_type='standard', epochs=20, batch_size=128, 
                    use_attention=True, save_model=True):
    """
    Train CNN model on embeddings.
    
    Args:
        model_type: 'standard', 'simple', or 'deep'
        epochs: Number of training epochs
        batch_size: Batch size for training
        use_attention: Use attention-weighted embeddings
        save_model: Save trained model
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load embeddings
    X_train, X_test, y_train, y_test = load_embeddings(use_attention=use_attention)
    
    # Convert to tensors
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Split train into train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")
    
    # Create model
    print(f"\nCreating {model_type} CNN model...")
    
    if model_type == 'simple':
        model = SimpleCNN1D(input_dim=2504, num_classes=2)
    elif model_type == 'deep':
        model = DeepCNN1D(input_dim=2504, num_classes=2)
    else:  # standard
        model = CNN1D(
            input_dim=2504,
            num_classes=2,
            conv_channels=(32, 64, 128, 256),
            kernel_sizes=(5, 5, 3, 3),
            pool_sizes=(2, 2, 2, 2),
            dropout_rate=0.3,
            fc_dims=(512, 256),
            fc_dropout=0.4
        )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    
    # Train
    trainer = CNNTrainer(model, device=device, learning_rate=0.001)
    history = trainer.train(train_loader, val_loader, epochs=epochs, patience=5)
    
    # Evaluate on test set
    print(f"\n{'=' * 80}")
    print(f"EVALUATING ON TEST SET")
    print(f"{'=' * 80}\n")
    
    test_metrics = trainer.evaluate(test_loader)
    
    print(f"Test Accuracy:  {test_metrics['accuracy']:.2f}%")
    print(f"Precision:      {test_metrics['precision']:.4f}")
    print(f"Recall:         {test_metrics['recall']:.4f}")
    print(f"F1 Score:       {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:        {test_metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(test_metrics['confusion_matrix'])
    
    # Save model and results
    if save_model:
        print(f"\n{'=' * 80}")
        print(f"SAVING MODEL AND RESULTS")
        print(f"{'=' * 80}\n")
        
        model_dir = './models'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f'cnn_{model_type}_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': model_type,
            'config': model.get_config() if hasattr(model, 'get_config') else None,
            'test_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in test_metrics.items() if k not in ['confusion_matrix', 
                                                                          'predictions', 
                                                                          'true_labels',
                                                                          'probabilities']},
        }, model_path)
        print(f"✓ {model_path}")
        
        # Save metrics
        metrics_path = os.path.join(model_dir, f'cnn_{model_type}_metrics.npy')
        np.save(metrics_path, test_metrics, allow_pickle=True)
        print(f"✓ {metrics_path}")
        
        # Save predictions
        pred_path = os.path.join(model_dir, f'cnn_{model_type}_predictions.npy')
        np.save(pred_path, {
            'predictions': test_metrics['predictions'],
            'true_labels': test_metrics['true_labels'],
            'probabilities': test_metrics['probabilities']
        }, allow_pickle=True)
        print(f"✓ {pred_path}")
    
    return model, trainer, test_metrics


if __name__ == "__main__":
    # Train standard CNN
    model, trainer, metrics = train_cnn_model(
        model_type='standard',
        epochs=20,
        batch_size=128,
        use_attention=True,
        save_model=True
    )
