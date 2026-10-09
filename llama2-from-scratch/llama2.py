import torch
import torch.nn as nn
import sentencepiece as spm


class LlamaTokenizer:
    def __init__(self, tokenizer_file):
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_file)
        self.tokenizer = sp

    def encode(self, text):
        return self.tokenizer.encode(text, out_type=int)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.eps = eps

    def forward(self, x):
        rms_x = (torch.mean(x**2, dim=-1, keepdim=True) + self.eps) ** 0.5
        return ((x / rms_x) * self.scale).to(dtype=x.dtype)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dtype):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)

def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dtype=None, device=None):
        super().__init__()

        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_in = d_in        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = self.d_out // self.num_heads
        self.context_length = context_length

        self.W_Q = nn.Linear(in_features = self.d_in, out_features = self.d_out, bias=False, dtype=dtype, device = self.device)
        self.W_K = nn.Linear(in_features = self.d_in, out_features = self.d_out, bias=False, dtype=dtype, device = self.device)
        self.W_V = nn.Linear(in_features = self.d_in, out_features = self.d_out, bias=False, dtype=dtype, device = self.device)

        mask = torch.triu(torch.full((self.d_in, self.d_in), -torch.inf), diagonal=1)
        self.register_buffer("mask", mask)

        # KV cache
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0

        # Optional linear projection layer to combine head outputs
        self.lin_proj = nn.Linear(in_features=self.d_out, out_features=d_out, bias=False, dtype=dtype)

        # Precomputed cos, sin for RoPE rotation
        cos, sin = precompute_rope_params(self.head_dim, context_length=self.context_length)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, inputs: torch.Tensor, use_cache=False) -> torch.Tensor:
        b, num_tokens, d_in = inputs.shape

        queries = self.W_Q(inputs)
        keys_new = self.W_K(inputs)
        values_new = self.W_V(inputs)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            keys, values = self.cache_k, self.cache_v
        else:
            keys, values = keys_new, values_new

        queries = queries.transpose(1, 2) # (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        queries = compute_rope(queries, self.cos, self.sin)
        keys = compute_rope(keys, self.cos, self.sin)

        # Attention scores
        attn_scores = queries @ keys.mT

        # Mask
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        if use_cache:
            mask = self.mask[self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K]
            self.ptr_current_pos += num_tokens_Q
        else:
            mask = self.mask[:num_tokens_Q, :num_tokens_K]
        masked_attn_scores = attn_scores + mask

        # Attention weights
        attn_weights = torch.softmax(masked_attn_scores / keys.shape[-1]**0.5, dim=-1)

        # Context vector
        context_vec = attn_weights @ values
        context_vec = context_vec.transpose(1, 2) # (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # Optional linear projection
        context_vec = self.lin_proj(context_vec)
        return context_vec

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, hidden_dim, context_length, num_heads, dtype, device:str = None):
        super().__init__()

        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if emb_dim % num_heads != 0:
            print("emb_dim must be divisible by num_heads")
            return

        self.norm1 = RMSNorm(emb_dim)
        self.norm2 = RMSNorm(emb_dim)
        self.att = MultiHeadAttention(emb_dim, emb_dim, context_length, num_heads, dtype, device)
        self.ff = FeedForward(emb_dim, hidden_dim, dtype)
        
    def forward(self, x, use_cache=False):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache)   # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x

class Llama2(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, context_length, num_heads, num_layers, dtype, device:str = None):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, emb_dim, dtype=dtype, device=device)

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(emb_dim, hidden_dim, context_length, num_heads, dtype, device)
            for _ in range(num_layers)]
        )

        self.final_norm = RMSNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False, dtype=dtype, device=device)
        
    def forward(self, in_idx, use_cache=False):
        x = self.tok_emb(in_idx)
        
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits

    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None, use_cache=True):
    
    if use_cache:
        model.reset_kv_cache()

    flag = False

    for _ in range(max_new_tokens):
        if use_cache and flag:
            with torch.inference_mode():
                logits = model(idx_next, use_cache)
        else:
            idx_cond = idx[:, -context_size:]
            flag = True
            with torch.inference_mode():
                logits = model(idx_cond, use_cache) # Shape: [b, num_tokens, vocab_size]
        logits = logits[:, -1, :] # only get logits of latest token. Shape: [b, vocab_size]

        # Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # numerical stability tip
            # subtract rowwise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

if __name__ == "__main__":
    LLAMA2_CONFIG_7B = {
        "vocab_size": 32000,     # Vocabulary size
        "context_length": 4096,  # Context length
        "emb_dim": 4096,         # Embedding dimension
        "num_heads": 32,           # Number of attention heads
        "num_layers": 32,          # Number of layers
        "hidden_dim": 11008,     # NEW: Size of the intermediate dimension in FeedForward
        "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
    }
    cfg = LLAMA2_CONFIG_7B

    model = Llama2(**cfg)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    from huggingface_hub import login
    import json

    with open("config.json", "r") as config_file:
        config = json.load(config_file)
        access_token = config["HF_ACCESS_TOKEN"]

    login(token=access_token)

    from huggingface_hub import hf_hub_download

    tokenizer_file = hf_hub_download(
        repo_id="meta-llama/Llama-2-7b",
        filename="tokenizer.model",
        local_dir="Llama-2-7b"
    )

    tokenizer = LlamaTokenizer(tokenizer_file)

    torch.manual_seed(123)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves", tokenizer).to(device),
        max_new_tokens=30,
        context_size=LLAMA2_CONFIG_7B["context_length"],
        top_k=1,
        temperature=0.,
        use_cache=True
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))