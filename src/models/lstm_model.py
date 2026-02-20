"""
lstm_model.py - Bidirectional LSTM with Attention for Text Classification
Architecture: Embedding → BiLSTM → Self-Attention → FC → Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    Allows the model to focus on relevant parts of the sequence.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(SelfAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) - 1 for valid tokens, 0 for padding
        Returns:
            output: (batch, seq_len, hidden_dim)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        residual = x

        # Project Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(context)

        # Residual + LayerNorm
        output = self.layer_norm(output + residual)

        return output, attention_weights


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM with Self-Attention for text classification.

    Architecture:
        Input → Embedding → Dropout
        → BiLSTM (multiple layers)
        → Self-Attention
        → Concat [forward, backward, attention_pooled]
        → FC layers → Softmax

    Captures both local and global dependencies in text.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        num_heads: int = 8,
        dropout_rate: float = 0.3,
        bidirectional: bool = True,
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        super(BiLSTMClassifier, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)

        self.embed_dropout = nn.Dropout(dropout_rate)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # Self-attention on LSTM outputs
        lstm_output_dim = hidden_dim * self.num_directions
        self.attention = SelfAttention(
            hidden_dim=lstm_output_dim,
            num_heads=num_heads,
            dropout=dropout_rate
        )

        # Classifier
        # We use: last hidden state (forward + backward) + attention pooled
        classifier_input_dim = lstm_output_dim * 2  # concat avg + max of attention output

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM weights with orthogonal initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1 (helps with gradient flow)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> dict:
        """
        Forward pass.
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len) - optional padding mask
        Returns:
            dict with 'logits', 'probabilities', 'attention_weights'
        """
        batch_size = input_ids.shape[0]

        # Embedding + dropout
        x = self.embedding(input_ids)   # (batch, seq_len, embed_dim)
        x = self.embed_dropout(x)

        # BiLSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_dim * num_directions)

        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, mask=attention_mask)
        # attn_out: (batch, seq_len, hidden_dim * num_directions)

        # Global pooling (avg + max) for fixed-size representation
        if attention_mask is not None:
            # Mask padding tokens before pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()
            attn_out_masked = attn_out * mask_expanded
            # Average over non-padding tokens
            avg_pool = attn_out_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
            # Max pool (set padding to -inf)
            attn_out_for_max = attn_out.masked_fill(mask_expanded == 0, float('-inf'))
            max_pool = attn_out_for_max.max(dim=1).values
        else:
            avg_pool = attn_out.mean(dim=1)
            max_pool = attn_out.max(dim=1).values

        # Concatenate pooled representations
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # Classification
        logits = self.classifier(pooled)

        return {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=-1),
            'attention_weights': attn_weights
        }

    def get_model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'model_name': 'BiLSTMClassifier',
            'total_parameters': total_params,
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional
        }


class HierarchicalLSTM(nn.Module):
    """
    Hierarchical LSTM: sentence-level → document-level.
    Best for long documents with multiple sentences.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        word_hidden_dim: int = 128,
        sent_hidden_dim: int = 128,
        num_classes: int = 3,
        dropout_rate: float = 0.3
    ):
        super(HierarchicalLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)

        # Word-level BiLSTM
        self.word_lstm = nn.LSTM(
            embed_dim, word_hidden_dim,
            bidirectional=True, batch_first=True
        )
        # Sentence-level BiLSTM
        self.sent_lstm = nn.LSTM(
            word_hidden_dim * 2, sent_hidden_dim,
            bidirectional=True, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(sent_hidden_dim * 2, num_classes)
        )

    def encode_sentence(self, word_ids: torch.Tensor) -> torch.Tensor:
        """Encode a single sentence to a vector."""
        x = self.embedding(word_ids)
        x = self.dropout(x)
        out, (h, _) = self.word_lstm(x)
        # Concat last forward and backward hidden states
        sent_vec = torch.cat([h[-2], h[-1]], dim=-1)
        return sent_vec

    def forward(self, input_ids: torch.Tensor, **kwargs) -> dict:
        """
        Args:
            input_ids: (batch, num_sentences, words_per_sentence)
        """
        if input_ids.dim() == 2:
            # If flat input, treat entire sequence as one sentence
            input_ids = input_ids.unsqueeze(1)

        batch_size, num_sents, seq_len = input_ids.shape

        # Encode each sentence
        sent_vecs = []
        for i in range(num_sents):
            sent_vec = self.encode_sentence(input_ids[:, i, :])
            sent_vecs.append(sent_vec)

        sent_matrix = torch.stack(sent_vecs, dim=1)  # (batch, num_sents, word_hidden*2)

        # Document-level encoding
        doc_out, (h, _) = self.sent_lstm(sent_matrix)
        doc_vec = torch.cat([h[-2], h[-1]], dim=-1)

        logits = self.classifier(doc_vec)
        return {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=-1)
        }


if __name__ == "__main__":
    batch_size, seq_len = 4, 128
    vocab_size, num_classes = 10000, 3

    model = BiLSTMClassifier(vocab_size=vocab_size, num_classes=num_classes)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)

    output = model(x, attention_mask=mask)
    print(f"✅ BiLSTM output shape: {output['logits'].shape}")
    print(f"   Attention weights shape: {output['attention_weights'].shape}")
    print(f"   Model info: {model.get_model_info()}")
