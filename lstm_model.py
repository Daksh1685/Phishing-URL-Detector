"""
PhishingLSTM Model for Sequential URL Analysis

Purpose: LSTM-based approach for phishing detection
Architecture: Embedding → LSTM → FC layers → Output
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class PhishingLSTM(nn.Module):
    """
    Standard LSTM model for phishing URL detection.
    
    Designed to capture sequential patterns in URL embeddings.
    Uses LSTM layers to understand temporal dependencies in URL structure.
    
    Architecture:
    - Input: 2,504D fused embedding
    - LSTM layers: 256 hidden units, 2 layers
    - Output: 2 classes (phishing/legitimate)
    """
    
    def __init__(self, input_dim=2504, hidden_dim=256, num_layers=2, dropout=0.3, num_classes=2):
        """
        Initialize PhishingLSTM.
        
        Args:
            input_dim (int): Input embedding dimension (default: 2,504D fused)
            hidden_dim (int): LSTM hidden dimension
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
            num_classes (int): Number of output classes
        """
        super(PhishingLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        logger.info(f"Initializing PhishingLSTM: input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")
        
        # Reshape embedding to sequence
        # Input shape: (batch_size, 2504) → (batch_size, seq_len, embedding_dim)
        self.seq_len = 4  # Reshape 2504 → (4, 626)
        self.embedding_dim = input_dim // self.seq_len
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)
        
        # Output Layer
        self.fc_out = nn.Linear(256, num_classes)
        
        self._init_weights()
        logger.info("PhishingLSTM initialized successfully")
    
    def _init_weights(self):
        """Initialize FC layer weights using Kaiming initialization."""
        for module in [self.fc1, self.fc2, self.fc_out]:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through LSTM.
        
        Args:
            x (Tensor): Input embedding of shape (batch_size, 2504)
        
        Returns:
            Tensor: Output logits of shape (batch_size, 2)
        """
        batch_size = x.size(0)
        
        # Reshape from (batch, 2504) → (batch, seq_len=4, embedding_dim=626)
        x = x.view(batch_size, self.seq_len, self.embedding_dim)
        
        # LSTM forward
        # Output shape: (batch, seq_len, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        # h_n shape: (num_layers, batch, hidden_dim)
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        
        # FC layers
        x = self.fc1(last_hidden)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        # Output
        logits = self.fc_out(x)
        
        return logits
    
    @staticmethod
    def test_phishing_lstm():
        """Test PhishingLSTM model."""
        logger.info("\n" + "="*60)
        logger.info("Testing PhishingLSTM")
        logger.info("="*60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        model = PhishingLSTM(input_dim=2504, hidden_dim=256, num_layers=2)
        model = model.to(device)
        
        # Test forward pass with different batch sizes
        for batch_size in [1, 4, 8, 32]:
            x = torch.randn(batch_size, 2504, device=device)
            output = model(x)
            
            assert output.shape == (batch_size, 2), f"Output shape mismatch: {output.shape}"
            assert not torch.isnan(output).any(), "NaN detected in output"
            assert not torch.isinf(output).any(), "Inf detected in output"
            
            logger.info(f"✅ Batch size {batch_size}: Output shape {output.shape}")
        
        # Test gradient flow
        logger.info("\nTesting gradient flow...")
        x = torch.randn(8, 2504, device=device, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, "No gradients computed"
        logger.info("✅ Gradient flow successful")
        
        logger.info("\n" + "="*60)
        logger.info("PhishingLSTM Test: PASSED ✅")
        logger.info("="*60 + "\n")
        
        return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    PhishingLSTM.test_phishing_lstm()
