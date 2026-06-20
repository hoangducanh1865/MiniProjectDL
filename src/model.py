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

    def __init__(self, input_dim: int, hidden_dims=(256, 128, 64),
                 dropout: float = 0.35, input_dropout: float = 0.05,
                 num_res_blocks: int = 2):
        super().__init__()

        self.input_dropout = nn.Dropout(input_dropout)

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
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(final_dim, dropout) for _ in range(num_res_blocks)
        ])

        self.head = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.encoder(self.input_dropout(x))
        h = self.res_blocks(h)
        return self.head(h).squeeze(-1)


class ICUMortalityTransformer(nn.Module):
    """
    FT-Transformer-style model for continuous tabular features.

    Each scalar feature is converted into a learned token, then a small
    Transformer encoder models feature-feature interactions.
    """

    def __init__(
        self,
        input_dim: int,
        d_token: int = 96,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.15,
        ff_multiplier: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_token = d_token

        self.feature_weight = nn.Parameter(torch.empty(input_dim, d_token))
        self.feature_bias = nn.Parameter(torch.zeros(input_dim, d_token))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.feature_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=num_heads,
            dim_feedforward=d_token * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.feature_weight)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0)
        tokens = tokens + self.feature_bias.unsqueeze(0)
        tokens = self.feature_dropout(tokens)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        h = torch.cat([cls, tokens], dim=1)
        h = self.encoder(h)
        h = self.norm(h[:, 0])
        return self.head(h).squeeze(-1)


def build_model(
    model_type: str,
    input_dim: int,
    hidden_dims=(256, 128, 64),
    dropout: float = 0.35,
    input_dropout: float = 0.05,
    num_res_blocks: int = 2,
    transformer_dim: int = 96,
    transformer_layers: int = 4,
    transformer_heads: int = 8,
):
    if model_type == "transformer":
        return ICUMortalityTransformer(
            input_dim=input_dim,
            d_token=transformer_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dropout=dropout,
        )
    return ICUMortalityMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        input_dropout=input_dropout,
        num_res_blocks=num_res_blocks,
    )
