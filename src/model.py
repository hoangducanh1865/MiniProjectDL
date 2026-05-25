import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.gelu(x + self.block(x)))


class ICUMortalityMLP(nn.Module):
    """
    Deep residual MLP for tabular ICU data.
    Architecture inspired by TabNet / ResNet-style tabular models.
    """

    def __init__(self, input_dim: int, hidden_dims=(512, 256, 128),
                 dropout: float = 0.3):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dims[0]),
                  nn.LayerNorm(hidden_dims[0]),
                  nn.GELU(),
                  nn.Dropout(dropout)]

        for i in range(len(hidden_dims) - 1):
            in_d, out_d = hidden_dims[i], hidden_dims[i + 1]
            layers += [
                nn.Linear(in_d, out_d),
                nn.LayerNorm(out_d),
                nn.GELU(),
                nn.Dropout(dropout),
            ]

        self.encoder = nn.Sequential(*layers)

        # Residual blocks at the final dimension
        final_dim = hidden_dims[-1]
        self.res_blocks = nn.Sequential(
            ResidualBlock(final_dim, dropout),
            ResidualBlock(final_dim, dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.encoder(x)
        h = self.res_blocks(h)
        return self.head(h).squeeze(-1)
