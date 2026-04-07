import torch
import torch.nn as nn


class ConvBranch(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChannelSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        weights = self.pool(x).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1, 1)
        return x * weights


class FreqAttention(nn.Module):
    """
    Yüksek frekans bölgelerine modelin daha fazla odaklanmasını sağlayan 
    öğrenilebilir frekans-bazlı dikkat mekanizması.
    """
    def __init__(self, num_freqs: int = 513):
        super().__init__()
        # Başlangıçta yüksek frekanslara daha fazla ağırlık verecek lineer bir maske (sigmoid üzerinden geçecek)
        init_weights = torch.linspace(-2.0, 2.0, num_freqs).view(1, 1, num_freqs, 1)
        self.freq_weights = nn.Parameter(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Channels, Freqs, Time)
        return x * torch.sigmoid(self.freq_weights)


class ArtifactNet(nn.Module):
    def __init__(self, num_classes: int = 2, base_channels: int = 16, num_freqs: int = 513):
        super().__init__()
        self.freq_attn = FreqAttention(num_freqs=num_freqs)
        self.raw_branch = ConvBranch(in_channels=1, base_channels=base_channels)
        self.residual_branch = ConvBranch(in_channels=1, base_channels=base_channels)

        fusion_channels = base_channels * 8
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 8, fusion_channels, kernel_size=1),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
        )
        self.se = ChannelSE(fusion_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fusion_channels * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def build_residual(x: torch.Tensor) -> torch.Tensor:
        time_delta = x[:, :, :, 1:] - x[:, :, :, :-1]
        freq_delta = x[:, :, 1:, :] - x[:, :, :-1, :]

        time_delta = nn.functional.pad(time_delta, (1, 0, 0, 0))
        freq_delta = nn.functional.pad(freq_delta, (0, 0, 1, 0))

        residual = 0.5 * (time_delta.abs() + freq_delta.abs())
        return residual

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.build_residual(x)
        x_attn = self.freq_attn(x)
        residual_attn = self.freq_attn(residual)
        
        raw_feat = self.raw_branch(x_attn)
        residual_feat = self.residual_branch(residual_attn)

        fused = torch.cat([raw_feat, residual_feat], dim=1)
        fused = self.fusion(fused)
        fused = self.se(fused)
        return fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.forward_features(x)
        return self.classifier(fused)
