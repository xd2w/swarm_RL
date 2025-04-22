import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LiquidNeuronLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dt=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dt = dt  # Euler integration step size

        # Learnable weights
        self.Wx = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.Wh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(hidden_dim))

        # Time constant parameters (input-dependent)
        self.tau_linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x_seq):
        """
        x_seq: (batch_size, seq_len, input_dim)
        Returns: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x_seq.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        outputs = []

        for t in range(seq_len):
            x_t = x_seq[:, t, :]  # (batch_size, input_dim)
            tau = F.softplus(self.tau_linear(x_t)) + 1e-2  # avoid div by 0

            # RNN-like update
            z = torch.tanh(F.linear(x_t, self.Wx) + F.linear(h, self.Wh) + self.b)
            dh = (-1.0 / tau) * (h - z)
            h = h + self.dt * dh  # Euler integration

            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1)


# 🧪 Test it on dummy data
if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    input_dim = 3
    hidden_dim = 8

    model = LiquidNeuronLayer(input_dim, hidden_dim)
    x = torch.randn(batch_size, seq_len, input_dim)
    y = model(x)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
