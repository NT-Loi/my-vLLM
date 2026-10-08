import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.eps = eps

    def forward(self, x):
        rms_x = (torch.mean(x**2, dim=-1, keepdim=True) + self.eps) ** 0.5
        return (x / rms_x) * self.scale

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False)
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

if __name__ == "__main__":
    torch.manual_seed(123)

    example_batch = torch.randn(2, 3, 4)

    rms_norm = RMSNorm(emb_dim=example_batch.shape[-1])
    rmsnorm_pytorch = torch.nn.RMSNorm(example_batch.shape[-1], eps=1e-5)

    print("My RMSNorm:", rms_norm(example_batch))
    print("PyTorch's RMSNorm:", rmsnorm_pytorch(example_batch))

    assert torch.allclose(rms_norm(example_batch), rmsnorm_pytorch(example_batch))

