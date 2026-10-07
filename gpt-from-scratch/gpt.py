from torch.cuda.green_contexts import _WORKQUEUE_SCOPE_VALUES
from os import W_OK
import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, qkv_bias=False, device:str = None):
        super().__init__()

        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.context_length = context_length
        self.d_in = d_in        
        self.d_out = d_out

        self.W_Q = nn.Linear(in_features = self.d_in, out_features = self.d_out, bias=qkv_bias, device = self.device)
        self.W_K = nn.Linear(in_features = self.d_in, out_features = self.d_out, bias=qkv_bias, device = self.device)
        self.W_V = nn.Linear(in_features = self.d_in, out_features = self.d_out, bias=qkv_bias, device = self.device)

        self.mask = torch.triu(torch.full((self.context_length, self.context_length), -torch.inf), diagonal=1)

    def forward(self, inputs: torch.Tensor):
        # inputs shape: (B, num_tokens, d_in)
        if inputs.shape[-2] > self.context_length:
            print("Input length exceeds maximum supported context length.")
            return None

        queries = self.W_Q(inputs)
        keys = self.W_K(inputs)
        values = self.W_V(inputs)

        # Attention scores
        attn_scores = queries @ keys.mT
        masked_attn_scores = attn_scores + self.mask[:inputs.shape[-2], :inputs.shape[-2]]

        # Attention weights
        masked_attn_weights = torch.softmax(masked_attn_scores / self.W_K.out_features**0.5, dim=-1)

        # Context vector
        context_vec = masked_attn_weights @ values
        return context_vec 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, device=None):
        super().__init__()

        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if d_out % num_heads != 0:
            print("d_out must be divisible by num_heads")
            return 

        self.d_in = d_in        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = self.d_out // self.num_heads

        self.W_Q = nn.Linear(in_features = self.d_in, out_features = self.d_out, bias=qkv_bias, device = self.device)
        self.W_K = nn.Linear(in_features = self.d_in, out_features = self.d_out, bias=qkv_bias, device = self.device)
        self.W_V = nn.Linear(in_features = self.d_in, out_features = self.d_out, bias=qkv_bias, device = self.device)

        self.mask = torch.triu(torch.full((self.d_in, self.d_in), -torch.inf), diagonal=1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Optional linear projection layer to combine head outputs
        self.lin_proj = nn.Linear(in_features=self.d_out, out_features=d_out, bias=True, device=self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = inputs.shape

        queries = self.W_Q(inputs)
        keys = self.W_K(inputs)
        values = self.W_V(inputs)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view_as(queries)
        values = values.view_as(queries)

        queries = queries.transpose(1, 2) # (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Attention scores
        attn_scores = queries @ keys.mT

        # Mask
        masked_attn_scores = attn_scores + self.mask[:num_tokens, :num_tokens]

        # Attention weights
        attn_weights = torch.softmax(masked_attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context vector
        context_vec = attn_weights @ values
        context_vec = context_vec.transpose(1, 2) # (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # Optional linear projection
        context_vec = self.lin_proj(context_vec)
        return context_vec


class GPT(nn.Module):
    def __init__(self):
        super.__init__()

    def forward(self, ):
        pass
if __name__ == "__main__":
    torch.manual_seed(42)
    inputs = torch.rand(3, 10, 10) 
    # ca = CausalAttention(d_in=inputs.shape[-1], d_out=20, context_length=10)
    # outputs = ca(inputs)
    # print(outputs.shape)

    mha = MultiHeadAttention(d_in=inputs.shape[-1], d_out=20, context_length=10, dropout=0.3, num_heads=5)
    outputs = mha(inputs)
    print(outputs.shape)