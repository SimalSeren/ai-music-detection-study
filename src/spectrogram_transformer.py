import torch
import torch.nn as nn


class SpectrogramTransformer(nn.Module):
    """
    AST benzeri hafif spectrogram transformer.
    Spectrogram crop'larini patch'lere bolup global self-attention ile siniflandirir.
    """

    def __init__(
        self,
        num_classes: int = 2,
        input_size: tuple[int, int] = (513, 64),
        patch_size: tuple[int, int] = (16, 16),
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size

        freq_bins, time_bins = input_size
        patch_freq, patch_time = patch_size
        grid_freq = freq_bins // patch_freq
        grid_time = time_bins // patch_time
        self.num_patches = grid_freq * grid_time

        self.patch_embed = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def _tokens_from_input(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(x)
        patches = patches.flatten(2).transpose(1, 2)
        batch_size = patches.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        return self.dropout(tokens)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens_from_input(x)
        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)
        return encoded[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.classifier(features)
