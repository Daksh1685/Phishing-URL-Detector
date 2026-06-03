"""
Chaotic LSTM Model with Logistic Map Initialization

Purpose: LSTM with chaos-based weight initialization for enhanced robustness
Architecture: Chaotic Init + LSTM → FC layers → Output
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

# Handle imports for chaotic_init
sys.path.insert(0, str(Path(__file__).parent))
try:
    from chaotic_init import logistic_map, chaotic_init, apply_chaotic_init
except ImportError:
    logger.warning("Could not import chaotic_init, will attempt fallback")
    chaotic_init = None


class ChaoticLSTM(nn.Module):
    """
    LSTM model with chaotic weight initialization.
    
    Combines LSTM architecture with logistic map-based weight initialization
    for enhanced robustness and better exploration of weight space.
    
    Architecture:
    - Input: 2,504D fused embedding
    - Chaotic Init: Logistic map (r=3.99) for LSTM weights
    - LSTM layers: 256 hidden units, 2 layers
    - Optional: Chaotic perturbation layer during training
    - Output: 2 classes (phishing/legitimate)
    """
    
    def __init__(self, input_dim=2504, hidden_dim=256, num_layers=2, dropout=0.3, 
                 num_classes=2, use_chaos=True, use_chaotic_layer=False, 
                 r=3.99, x0=0.5, epsilon=0.005):
        """
        Initialize ChaoticLSTM.
        
        Args:
            input_dim (int): Input embedding dimension
            hidden_dim (int): LSTM hidden dimension
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
            num_classes (int): Number of output classes
            use_chaos (bool): Enable chaotic weight initialization
            use_chaotic_layer (bool): Enable chaotic perturbation layer
            r (float): Logistic map bifurcation parameter (default: 3.99)
            x0 (float): Initial condition for logistic map (default: 0.5)
            epsilon (float): Perturbation magnitude (default: 0.005)
        """
        super(ChaoticLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_chaos = use_chaos
        self.use_chaotic_layer = use_chaotic_layer
        self.r = r
        self.x0 = x0
        self.epsilon = epsilon
        
        logger.info(f"Initializing ChaoticLSTM: use_chaos={use_chaos}, r={r}, epsilon={epsilon}")
        
        # Reshape embedding to sequence
        self.seq_len = 4
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
        
        # Apply chaotic initialization if enabled
        if self.use_chaos:
            self._apply_chaotic_initialization()
        else:
            self._init_weights()
        
        logger.info("ChaoticLSTM initialized successfully")
    
    def _init_weights(self):
        """Standard Kaiming initialization for FC layers."""
        for module in [self.fc1, self.fc2, self.fc_out]:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _apply_chaotic_initialization(self):
        """
        Apply chaotic weight initialization to all layers.
        
        Uses logistic map (r=3.99) for weight initialization to improve
        exploration of weight space during training.
        """
        logger.info("\n" + "="*60)
        logger.info("APPLYING CHAOTIC INITIALIZATION TO LSTM")
        logger.info("="*60)
        
        try:
            # Apply to FC layers
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    logger.info(f"\nChaotic Init {name}: {module.weight.shape}")
                    
                    # Generate chaotic weights
                    with torch.no_grad():
                        # Logistic map
                        chaos_sequence = self._logistic_map_tensor(
                            torch.zeros(100),
                            iterations=100
                        )
                        
                        # Initialize weights
                        fan_in = module.weight.size(1)
                        scale = 1.0 / (fan_in ** 0.5)
                        
                        # Sample from chaos sequence and scale
                        chaos_init_weights = []
                        for i in range(module.weight.numel()):
                            idx = i % len(chaos_sequence)
                            chaos_init_weights.append(chaos_sequence[idx].item())
                        
                        chaos_init_weights = torch.tensor(chaos_init_weights, dtype=module.weight.dtype)
                        chaos_init_weights = chaos_init_weights * scale
                        module.weight.copy_(chaos_init_weights.reshape(module.weight.shape))
                        
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                        
                        logger.info(f"  Mean: {module.weight.mean():.6f}, Std: {module.weight.std():.6f}")
            
            # Apply to LSTM weights
            logger.info("\nChaotic Init LSTM layers...")
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    with torch.no_grad():
                        chaos_seq = self._logistic_map_tensor(torch.zeros(100), iterations=100)
                        fan_in = param.size(1) if len(param.shape) > 1 else param.size(0)
                        scale = 1.0 / (fan_in ** 0.5)
                        
                        chaos_init = []
                        for i in range(param.numel()):
                            idx = i % len(chaos_seq)
                            chaos_init.append(chaos_seq[idx].item())
                        
                        chaos_init = torch.tensor(chaos_init, dtype=param.dtype)
                        chaos_init = chaos_init * scale
                        param.copy_(chaos_init.reshape(param.shape))
            
            logger.info("\n" + "="*60)
            logger.info("CHAOTIC INITIALIZATION COMPLETE ✅")
            logger.info("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"Error during chaotic initialization: {e}")
            logger.warning("Falling back to standard initialization")
            self._init_weights()
    
    def _logistic_map_tensor(self, x, iterations=100):
        """
        Generate logistic map sequence as tensor.
        
        Args:
            x (Tensor): Starting tensor
            iterations (int): Number of iterations
        
        Returns:
            Tensor: Chaotic sequence
        """
        device = x.device
        r = torch.tensor(self.r, dtype=torch.float32, device=device)
        x0 = torch.tensor(self.x0, dtype=torch.float32, device=device)
        
        # Generate sequence
        sequence = []
        x_curr = x0
        
        # Skip transients (first 20 iterations)
        for _ in range(20):
            x_curr = r * x_curr * (1 - x_curr)
        
        # Collect sequence
        for _ in range(iterations):
            x_curr = r * x_curr * (1 - x_curr)
            sequence.append(x_curr)
        
        return torch.stack(sequence)
    
    def forward(self, x, return_perturbation=False):
        """
        Forward pass through Chaotic LSTM.
        
        Args:
            x (Tensor): Input embedding of shape (batch_size, 2504)
            return_perturbation (bool): Whether to return perturbation magnitude
        
        Returns:
            Tensor: Output logits of shape (batch_size, 2)
            or tuple: (logits, perturbation_magnitude) if return_perturbation=True
        """
        batch_size = x.size(0)
        
        # Reshape from (batch, 2504) → (batch, seq_len=4, embedding_dim=626)
        x = x.view(batch_size, self.seq_len, self.embedding_dim)
        
        # Optional chaotic perturbation during training
        if self.use_chaotic_layer and self.training:
            with torch.no_grad():
                perturbation = self._generate_chaotic_perturbation(x)
            x = x + perturbation
            perturbation_magnitude = perturbation.abs().max().item()
        else:
            perturbation_magnitude = 0.0
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
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
        
        if return_perturbation:
            return logits, perturbation_magnitude
        return logits
    
    def _generate_chaotic_perturbation(self, x):
        """Generate chaotic perturbation for input."""
        # Normalize input to [0, 1]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        # Apply logistic map
        r = torch.tensor(self.r, dtype=x.dtype, device=x.device)
        x_chaos = r * x_norm * (1 - x_norm)
        
        # Scale to [-1, 1]
        x_chaos = 2 * x_chaos - 1
        
        # Apply epsilon scaling and clamp
        perturbation = self.epsilon * x_chaos
        perturbation = torch.clamp(perturbation, -1e6, 1e6)
        
        return perturbation
    
    def get_chaos_config(self):
        """Return current chaos configuration."""
        return {
            'use_chaos': self.use_chaos,
            'use_chaotic_layer': self.use_chaotic_layer,
            'r': self.r,
            'x0': self.x0,
            'epsilon': self.epsilon
        }
    
    @staticmethod
    def test_chaotic_lstm():
        """Test ChaoticLSTM model."""
        logger.info("\n" + "="*60)
        logger.info("Testing ChaoticLSTM")
        logger.info("="*60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test with chaos enabled
        model = ChaoticLSTM(input_dim=2504, use_chaos=True, use_chaotic_layer=False)
        model = model.to(device)
        model.eval()
        
        for batch_size in [1, 4, 8, 32]:
            x = torch.randn(batch_size, 2504, device=device)
            output = model(x)
            
            assert output.shape == (batch_size, 2), f"Output shape mismatch: {output.shape}"
            assert not torch.isnan(output).any(), "NaN detected"
            assert not torch.isinf(output).any(), "Inf detected"
            
            logger.info(f"✅ Batch {batch_size}: Output {output.shape}")
        
        # Test with perturbation layer
        model_with_layer = ChaoticLSTM(input_dim=2504, use_chaos=True, use_chaotic_layer=True)
        model_with_layer = model_with_layer.to(device)
        model_with_layer.train()
        
        x = torch.randn(8, 2504, device=device)
        output, pert_mag = model_with_layer(x, return_perturbation=True)
        
        logger.info(f"✅ With perturbation layer: Magnitude {pert_mag:.6f}")
        
        # Test gradient flow
        logger.info("\nTesting gradient flow...")
        x = torch.randn(8, 2504, device=device, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, "No gradients"
        logger.info("✅ Gradient flow successful")
        
        logger.info("\n" + "="*60)
        logger.info("ChaoticLSTM Test: PASSED ✅")
        logger.info("="*60 + "\n")
        
        return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ChaoticLSTM.test_chaotic_lstm()
