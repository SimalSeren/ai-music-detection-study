import torch
import torch.nn as nn

from resnet_spectrogram import ResNetSpectrogram


class CropEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.backbone = ResNetSpectrogram(num_classes=embedding_dim, base_channels=32)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        return self.projection(features)


class AttentionMILClassifier(nn.Module):
    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.encoder = CropEncoder(embedding_dim=embedding_dim)
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, crops: torch.Tensor, crop_mask: torch.Tensor):
        batch_size, n_crops, channels, height, width = crops.shape
        flat_crops = crops.view(batch_size * n_crops, channels, height, width)
        embeddings = self.encoder(flat_crops)
        embeddings = embeddings.view(batch_size, n_crops, -1)

        attn_logits = self.attention(embeddings).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~crop_mask, float("-inf"))
        attn_weights = torch.softmax(attn_logits, dim=1)

        pooled = torch.sum(embeddings * attn_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(pooled)
        return logits, attn_weights, embeddings
