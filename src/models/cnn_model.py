"""
cnn_model.py - Convolutional Neural Network for Text Classification
Architecture: Embedding → Multiple Conv Filters → MaxPool → FC → Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class TextCNN(nn.Module):
    """
    Kim (2014) style TextCNN for sentence classification.
    Uses multiple filter sizes to capture different n-gram features.

    Architecture:
        Input (batch, seq_len) → Embedding (batch, seq_len, embed_dim)
        → Conv1D with multiple filter sizes → MaxPool over time
        → Concat → Dropout → FC → Softmax
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_classes: int = 3,
        num_filters: int = 100,
        filter_sizes: List[int] = [2, 3, 4, 5],
        dropout_rate: float = 0.5,
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        super(TextCNN, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)

        # Parallel convolutional layers with different filter sizes
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=fs
                ),
                nn.BatchNorm1d(num_filters),
                nn.ReLU()
            )
            for fs in filter_sizes
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected output layer
        total_filters = num_filters * len(filter_sizes)
        self.fc1 = nn.Linear(total_filters, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for FC layers."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> dict:
        """
        Forward pass.
        Args:
            input_ids: (batch_size, seq_len)
        Returns:
            dict with 'logits' and 'probabilities'
        """
        # Embedding: (batch, seq_len) → (batch, seq_len, embed_dim)
        x = self.embedding(input_ids)

        # Transpose for Conv1d: (batch, embed_dim, seq_len)
        x = x.permute(0, 2, 1)

        # Apply each conv filter and max-pool
        pooled_outputs = []
        for conv in self.conv_layers:
            # Conv: (batch, num_filters, seq_len - filter_size + 1)
            conv_out = conv(x)
            # Max-over-time pooling: (batch, num_filters)
            pooled = F.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2)
            pooled_outputs.append(pooled)

        # Concatenate all filter outputs: (batch, num_filters * len(filter_sizes))
        x = torch.cat(pooled_outputs, dim=1)

        # Dropout + FC layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        return {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=-1)
        }

    def get_model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'model_name': 'TextCNN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'filter_sizes': self.filter_sizes,
            'num_filters': self.num_filters,
            'num_classes': self.num_classes
        }


class DeepTextCNN(nn.Module):
    """
    Deeper CNN variant with residual connections.
    Better for longer documents.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_classes: int = 3,
        num_filters: int = 128,
        dropout_rate: float = 0.3
    ):
        super(DeepTextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Block 1: bigrams
        self.conv_block1 = self._make_conv_block(embed_dim, num_filters, kernel_size=2)

        # Block 2: trigrams
        self.conv_block2 = self._make_conv_block(num_filters, num_filters, kernel_size=3)

        # Block 3: 4-grams with residual
        self.conv_block3 = self._make_conv_block(num_filters, num_filters, kernel_size=4)

        # Residual projection
        self.residual_proj = nn.Conv1d(embed_dim, num_filters, kernel_size=1)

        self.dropout = nn.Dropout(dropout_rate)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Output: avg_pool + max_pool = num_filters * 2
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def _make_conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, input_ids: torch.Tensor, **kwargs) -> dict:
        # Embedding
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)         # (batch, embed_dim, seq_len)

        # Residual path
        residual = self.residual_proj(x)

        # Main path through conv blocks
        x = self.conv_block1(x)

        # Handle size mismatch for conv blocks with different kernel sizes
        min_len = min(x.shape[2], residual.shape[2])
        x = x[:, :, :min_len]
        residual = residual[:, :, :min_len]

        x = x + residual  # Residual connection

        # Global pooling (concat avg + max for richer representation)
        avg = self.global_avg_pool(x).squeeze(2)
        max_ = self.global_max_pool(x).squeeze(2)
        x = torch.cat([avg, max_], dim=1)

        x = self.dropout(x)
        logits = self.classifier(x)

        return {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=-1)
        }


if __name__ == "__main__":
    # Quick test
    batch_size, seq_len = 4, 128
    vocab_size, num_classes = 10000, 3

    model = TextCNN(vocab_size=vocab_size, num_classes=num_classes)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    output = model(x)
    print(f"✅ TextCNN output shape: {output['logits'].shape}")
    print(f"   Model info: {model.get_model_info()}")

    model2 = DeepTextCNN(vocab_size=vocab_size, num_classes=num_classes)
    output2 = model2(x)
    print(f"✅ DeepTextCNN output shape: {output2['logits'].shape}")
