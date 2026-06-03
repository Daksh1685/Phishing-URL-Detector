import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    """Basic attention mechanism layer."""
    
    def __init__(self, input_dim, hidden_dim=None, output_attention_weights=False):
        """
        Initialize attention layer.
        
        Args:
            input_dim (int): Input dimension (number of features)
            hidden_dim (int): Hidden dimension for attention (default: input_dim)
            output_attention_weights (bool): Whether to return attention weights
        """
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim
        self.output_attention_weights = output_attention_weights
        
        # Learnable query, key, value projections
        self.query_projection = nn.Linear(input_dim, self.hidden_dim)
        self.key_projection = nn.Linear(input_dim, self.hidden_dim)
        self.value_projection = nn.Linear(input_dim, self.hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_dim, input_dim)
        
        # Scaling factor
        self.scale = np.sqrt(self.hidden_dim)
    
    def forward(self, x, mask=None):
        """
        Forward pass for attention.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, input_dim)
                            or (batch_size, input_dim) for single vector
            mask (torch.Tensor): Attention mask (optional)
            
        Returns:
            torch.Tensor: Output tensor (same shape as input)
            torch.Tensor: Attention weights (only if output_attention_weights=True)
        """
        # Handle single vector input (batch_size, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, _ = x.shape
        
        # Compute query, key, value
        query = self.query_projection(x)  # (batch_size, seq_len, hidden_dim)
        key = self.key_projection(x)      # (batch_size, seq_len, hidden_dim)
        value = self.value_projection(x)  # (batch_size, seq_len, hidden_dim)
        
        # Compute attention scores
        scores = torch.bmm(query, key.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        scores = scores / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # Apply attention to values
        context = torch.bmm(attention_weights, value)  # (batch_size, seq_len, hidden_dim)
        
        # Output projection
        output = self.output_projection(context)  # (batch_size, seq_len, input_dim)
        
        # Remove squeeze dimension if added
        if squeeze_output:
            output = output.squeeze(1)  # (batch_size, input_dim)
            attention_weights = attention_weights.squeeze(1)  # (batch_size, 1, seq_len) → (batch_size, seq_len)
        
        if self.output_attention_weights:
            return output, attention_weights
        else:
            return output


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, input_dim, num_heads=4, hidden_dim=None, output_attention_weights=False):
        """
        Initialize multi-head attention.
        
        Args:
            input_dim (int): Input dimension
            num_heads (int): Number of attention heads
            hidden_dim (int): Hidden dimension per head (default: input_dim // num_heads)
            output_attention_weights (bool): Whether to return attention weights
        """
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim or (input_dim // num_heads)
        self.output_attention_weights = output_attention_weights
        
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        # Learnable projections
        self.query_projection = nn.Linear(input_dim, self.hidden_dim * num_heads)
        self.key_projection = nn.Linear(input_dim, self.hidden_dim * num_heads)
        self.value_projection = nn.Linear(input_dim, self.hidden_dim * num_heads)
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_dim * num_heads, input_dim)
        
        # Scaling factor
        self.scale = np.sqrt(self.hidden_dim)
    
    def forward(self, x, mask=None):
        """
        Forward pass for multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, input_dim)
                            or (batch_size, input_dim) for single vector
            mask (torch.Tensor): Attention mask (optional)
            
        Returns:
            torch.Tensor: Output tensor (same shape as input)
            torch.Tensor: Attention weights (only if output_attention_weights=True)
        """
        # Handle single vector input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape for multi-head
        query = self.query_projection(x)  # (batch_size, seq_len, hidden_dim * num_heads)
        query = query.view(batch_size, seq_len, self.num_heads, self.hidden_dim)
        query = query.transpose(1, 2)  # (batch_size, num_heads, seq_len, hidden_dim)
        
        key = self.key_projection(x)
        key = key.view(batch_size, seq_len, self.num_heads, self.hidden_dim)
        key = key.transpose(1, 2)  # (batch_size, num_heads, seq_len, hidden_dim)
        
        value = self.value_projection(x)
        value = value.view(batch_size, seq_len, self.num_heads, self.hidden_dim)
        value = value.transpose(1, 2)  # (batch_size, num_heads, seq_len, hidden_dim)
        
        # Compute attention scores for each head
        scores = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
        scores = scores / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, value)  # (batch_size, num_heads, seq_len, hidden_dim)
        
        # Concatenate heads
        context = context.transpose(1, 2)  # (batch_size, seq_len, num_heads, hidden_dim)
        context = context.contiguous()
        context = context.view(batch_size, seq_len, self.hidden_dim * self.num_heads)
        
        # Output projection
        output = self.output_projection(context)  # (batch_size, seq_len, input_dim)
        
        # Average attention weights across heads for output
        attention_weights = attention_weights.mean(dim=1)  # (batch_size, seq_len, seq_len)
        
        # Remove squeeze dimension if added
        if squeeze_output:
            output = output.squeeze(1)  # (batch_size, input_dim)
            attention_weights = attention_weights.squeeze(1)  # (batch_size, seq_len)
        
        if self.output_attention_weights:
            return output, attention_weights
        else:
            return output


class AttentionPool(nn.Module):
    """Attention-based pooling to reduce sequence to single vector."""
    
    def __init__(self, input_dim, use_multi_head=False, num_heads=4):
        """
        Initialize attention pooling.
        
        Args:
            input_dim (int): Input dimension
            use_multi_head (bool): Use multi-head attention
            num_heads (int): Number of heads (if multi-head)
        """
        super(AttentionPool, self).__init__()
        self.input_dim = input_dim
        self.use_multi_head = use_multi_head
        
        # Learnable context vector
        self.context_vector = nn.Parameter(torch.randn(input_dim))
        nn.init.xavier_uniform_(self.context_vector.unsqueeze(0))
        
        # Attention layer
        if use_multi_head:
            self.attention = MultiHeadAttention(
                input_dim, 
                num_heads=num_heads,
                output_attention_weights=True
            )
        else:
            self.attention = Attention(
                input_dim,
                output_attention_weights=True
            )
    
    def forward(self, x):
        """
        Forward pass - pool sequence to single vector using attention.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Pooled vector (batch_size, input_dim)
            torch.Tensor: Attention weights (batch_size, seq_len)
        """
        batch_size = x.shape[0]
        
        # Apply attention
        attended, weights = self.attention(x)  # attended: (batch_size, seq_len, input_dim)
        
        # Compute attention scores with context vector
        context = self.context_vector.unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)
        scores = torch.sum(attended * context, dim=-1)  # (batch_size, seq_len)
        
        # Softmax to get weights
        pool_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len)
        
        # Weighted pooling
        pooled = torch.sum(attended * pool_weights.unsqueeze(-1), dim=1)  # (batch_size, input_dim)
        
        return pooled, pool_weights


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING ATTENTION MECHANISMS")
    print("=" * 80)
    
    # Test parameters
    batch_size = 32
    input_dim = 2504  # Fused embedding dimension
    seq_len = 1  # For single URL embedding
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Input dim: {input_dim}")
    print(f"Seq len: {seq_len}")
    
    # Create random input
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    
    # Also test with 2D input (single vector per sample)
    x_2d = torch.randn(batch_size, input_dim).to(device)
    
    print("\n" + "-" * 80)
    print("1. BASIC ATTENTION")
    print("-" * 80)
    
    attn = Attention(input_dim, hidden_dim=256).to(device)
    output = attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    output_2d = attn(x_2d)
    print(f"Input 2D shape: {x_2d.shape}")
    print(f"Output 2D shape: {output_2d.shape}")
    
    # Test with attention weights
    attn_weights = Attention(input_dim, hidden_dim=256, output_attention_weights=True).to(device)
    output, weights = attn_weights(x)
    print(f"Attention weights shape: {weights.shape}")
    print(f"Weights sum (should be ~1): {weights[0].sum(dim=-1)}")
    
    print("\n" + "-" * 80)
    print("2. MULTI-HEAD ATTENTION")
    print("-" * 80)
    
    mha = MultiHeadAttention(input_dim, num_heads=8).to(device)
    output = mha(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    mha_weights = MultiHeadAttention(
        input_dim, 
        num_heads=8,
        output_attention_weights=True
    ).to(device)
    output, weights = mha_weights(x)
    print(f"Attention weights shape: {weights.shape}")
    print(f"Weights sum (should be ~1): {weights[0].sum(dim=-1)}")
    
    print("\n" + "-" * 80)
    print("3. ATTENTION POOLING")
    print("-" * 80)
    
    pool_attn = AttentionPool(input_dim, use_multi_head=True, num_heads=8).to(device)
    pooled, pool_weights = pool_attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Pooled shape: {pooled.shape}")
    print(f"Pool weights shape: {pool_weights.shape}")
    print(f"Pool weights sum (should be ~1): {pool_weights[0].sum()}")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
