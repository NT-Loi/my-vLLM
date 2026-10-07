from torch import device
import torch
import torch.nn as nn

import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "num_heads": 12,          # Number of attention heads
    "num_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

cfg = GPT_CONFIG_124M

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
        self.lin_proj = nn.Linear(in_features=self.d_out, out_features=d_out, bias=True)

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

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, context_length, num_heads, drop_rate, qkv_bias=False, device:str = None):
        super().__init__()

        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if emb_dim % num_heads != 0:
            print("emb_dim must be divisible by num_heads")
            return

        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.mha = MultiHeadAttention(emb_dim, emb_dim, context_length, drop_rate, num_heads, qkv_bias, device)
        self.drop_shortcut = nn.Dropout(drop_rate)
        self.ff = FeedForward(emb_dim)
        
    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.mha(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, emb_dim, context_length, num_heads, num_layers, drop_rate, qkv_bias=False, device:str = None):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.drop_emb = nn.Dropout(drop_rate)
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(emb_dim, context_length, num_heads, drop_rate, qkv_bias, device) 
            for _ in range(num_layers)])
        
        self.final_norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

if __name__ == "__main__":
    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []

    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)

    # ca = CausalAttention(d_in=batch.shape[-1], d_out=20, context_length=10)
    # outputs = ca(batch)
    # print(outputs.shape)

    # mha = MultiHeadAttention(d_in=batch.shape[-1], d_out=20, context_length=10, dropout=0.3, num_heads=5)
    # outputs = mha(batch)
    # print(outputs.shape)

    # trf = TransformerBlock(cfg["emb_dim"], cfg["context_length"], cfg["num_heads"], cfg["drop_rate"], cfg["qkv_bias"])
    # outputs = trf(batch)
    # print(outputs.shape)

    model = GPT(**cfg)
    # model.eval()
    # with torch.inference_mode():
    outputs = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", outputs.shape)
    print(outputs)