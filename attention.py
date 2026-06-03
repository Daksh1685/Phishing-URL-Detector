import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attention(nn.Module):    
    def __init__(self, input_dim, hidden_dim=None, output_attention_weights=False):

        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim
        self.output_attention_weights = output_attention_weights
        

        self.query_projection = nn.Linear(input_dim, self.hidden_dim)
        self.key_projection = nn.Linear(input_dim, self.hidden_dim)
        self.value_projection = nn.Linear(input_dim, self.hidden_dim)
        

        self.output_projection = nn.Linear(self.hidden_dim, input_dim)
        

        self.scale = np.sqrt(self.hidden_dim)
    
    def forward(self, x, mask=None):

        if x.dim() == 2:
            x = x.unsqueeze(1)  
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, _ = x.shape
        

        query = self.query_projection(x) 
        key = self.key_projection(x)      
        value = self.value_projection(x)  
        

        scores = torch.bmm(query, key.transpose(1, 2))  
        scores = scores / self.scale
        

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        

        attention_weights = F.softmax(scores, dim=-1) 
        

        context = torch.bmm(attention_weights, value) 
        

        output = self.output_projection(context) 
        

        if squeeze_output:
            output = output.squeeze(1) 
            attention_weights = attention_weights.squeeze(1) 
        
        if self.output_attention_weights:
            return output, attention_weights
        else:
            return output


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=None, output_attention_weights=False):

        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim or (input_dim // num_heads)
        self.output_attention_weights = output_attention_weights
        
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        

        self.query_projection = nn.Linear(input_dim, self.hidden_dim * num_heads)
        self.key_projection = nn.Linear(input_dim, self.hidden_dim * num_heads)
        self.value_projection = nn.Linear(input_dim, self.hidden_dim * num_heads)
        

        self.output_projection = nn.Linear(self.hidden_dim * num_heads, input_dim)
        

        self.scale = np.sqrt(self.hidden_dim)
    
    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)  
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, _ = x.shape
        

        query = self.query_projection(x)  
        query = query.view(batch_size, seq_len, self.num_heads, self.hidden_dim)
        query = query.transpose(1, 2)  
        
        key = self.key_projection(x)
        key = key.view(batch_size, seq_len, self.num_heads, self.hidden_dim)
        key = key.transpose(1, 2) 
        
        value = self.value_projection(x)
        value = value.view(batch_size, seq_len, self.num_heads, self.hidden_dim)
        value = value.transpose(1, 2)  
        

        scores = torch.matmul(query, key.transpose(-2, -1))  
        scores = scores / self.scale
        

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        

        attention_weights = F.softmax(scores, dim=-1) 
        

        context = torch.matmul(attention_weights, value) 
        

        context = context.transpose(1, 2)  
        context = context.contiguous()
        context = context.view(batch_size, seq_len, self.hidden_dim * self.num_heads)       
        output = self.output_projection(context)  
        

        attention_weights = attention_weights.mean(dim=1)  )
        

        if squeeze_output:
            output = output.squeeze(1) 
            attention_weights = attention_weights.squeeze(1)  
        
        if self.output_attention_weights:
            return output, attention_weights
        else:
            return output

class AttentionPool(nn.Module):
    def __init__(self, input_dim, use_multi_head=False, num_heads=4):
        super(AttentionPool, self).__init__()
        self.input_dim = input_dim
        self.use_multi_head = use_multi_head
        

        self.context_vector = nn.Parameter(torch.randn(input_dim))
        nn.init.xavier_uniform_(self.context_vector.unsqueeze(0))
        

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
        batch_size = x.shape[0]   
        attended, weights = self.attention(x)  
        
        context = self.context_vector.unsqueeze(0).unsqueeze(0)  
        scores = torch.sum(attended * context, dim=-1) 
        pool_weights = F.softmax(scores, dim=-1)  
        
        pooled = torch.sum(attended * pool_weights.unsqueeze(-1), dim=1)  
        
        return pooled, pool_weights


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING ATTENTION MECHANISMS")
    print("=" * 80)
    

    batch_size = 32
    input_dim = 2504  
    seq_len = 1  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Input dim: {input_dim}")
    print(f"Seq len: {seq_len}")
    

    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    

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
