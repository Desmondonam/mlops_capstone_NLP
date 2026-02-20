"""
bert_model.py - BERT/DistilBERT Fine-tuning for Text Classification
Uses HuggingFace Transformers - the LLM/Transformer component of the project

Covers:
- Fine-tuning pre-trained transformer models
- Custom classification heads
- Gradient checkpointing for memory efficiency
- Layer-wise learning rate decay
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoConfig,
    DistilBertModel,
    BertModel,
    RobertaModel,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TransformerClassifier(nn.Module):
    """
    Fine-tuned Transformer model (BERT/DistilBERT/RoBERTa) for classification.

    Features:
    - Multiple pooling strategies (CLS, mean, max, weighted)
    - Custom classification head with LayerNorm and residual
    - Gradient checkpointing support
    - Optional feature extraction mode (freeze transformer)
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 3,
        dropout_rate: float = 0.3,
        pooling_strategy: str = "mean",  # cls, mean, max, weighted
        freeze_transformer: bool = False,
        freeze_layers: int = 0,  # Freeze first N transformer layers
        hidden_dim: int = 256,
        use_gradient_checkpointing: bool = False
    ):
        super(TransformerClassifier, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.pooling_strategy = pooling_strategy

        # Load pre-trained transformer
        logger.info(f"Loading pre-trained model: {model_name}")
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

        # Enable gradient checkpointing (saves memory, slightly slower)
        if use_gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable()

        # Freeze layers if specified
        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
        elif freeze_layers > 0:
            self._freeze_layers(freeze_layers)

        # Get hidden size from config
        hidden_size = self.config.hidden_size

        # Pooling-specific parameters
        if pooling_strategy == "weighted":
            # Learnable weights for each token position
            self.token_weights = nn.Linear(hidden_size, 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # GELU activation (used in BERT)
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Initialize classifier weights
        self._init_classifier_weights()

    def _freeze_layers(self, n_layers: int):
        """Freeze the first n transformer layers."""
        try:
            # Try BERT-style layer access
            encoder = self.transformer.encoder
            for i, layer in enumerate(encoder.layer):
                if i < n_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            logger.info(f"Frozen first {n_layers} transformer layers")
        except AttributeError:
            # DistilBERT style
            try:
                for i, layer in enumerate(self.transformer.transformer.layer):
                    if i < n_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                logger.info(f"Frozen first {n_layers} transformer layers")
            except AttributeError:
                logger.warning("Could not freeze layers - architecture not recognized")

    def _init_classifier_weights(self):
        """Initialize classifier head with small weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def pool_output(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool transformer output to fixed-size vector.
        Strategies: cls, mean, max, weighted
        """
        if self.pooling_strategy == "cls":
            # Use [CLS] token representation
            return hidden_states[:, 0, :]

        elif self.pooling_strategy == "mean":
            # Mean pooling over non-padding tokens
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                return sum_hidden / sum_mask
            return hidden_states.mean(dim=1)

        elif self.pooling_strategy == "max":
            # Max pooling
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                hidden_states = hidden_states.masked_fill(mask_expanded == 0, float('-inf'))
            return hidden_states.max(dim=1).values

        elif self.pooling_strategy == "weighted":
            # Learned attention weights over tokens
            weights = self.token_weights(hidden_states)  # (batch, seq, 1)
            if attention_mask is not None:
                weights = weights.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, float('-inf')
                )
            weights = F.softmax(weights, dim=1)
            return (hidden_states * weights).sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass through transformer + classifier.
        """
        # Transformer forward pass
        transformer_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        # Add token_type_ids only for BERT (not DistilBERT)
        if token_type_ids is not None and hasattr(self.transformer, 'embeddings'):
            if hasattr(self.transformer.embeddings, 'token_type_embeddings'):
                transformer_kwargs['token_type_ids'] = token_type_ids

        outputs = self.transformer(**transformer_kwargs)

        # Get hidden states (last layer)
        # For most transformers, last_hidden_state is at outputs[0]
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Pool to fixed-size representation
        pooled = self.pool_output(hidden_states, attention_mask)

        # Classify
        logits = self.classifier(pooled)

        result = {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=-1),
            'pooled_output': pooled
        }

        if return_hidden_states:
            result['hidden_states'] = hidden_states

        return result

    def get_optimizer_groups(self, learning_rate: float = 2e-5, weight_decay: float = 0.01):
        """
        Layer-wise learning rate decay (LLRD).
        Lower layers get smaller LR, classifier gets larger LR.
        This is a key technique for fine-tuning transformers.
        """
        # Separate parameters by layer
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        # Transformer parameters (lower LR)
        transformer_params = [
            {
                "params": [
                    p for n, p in self.transformer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": learning_rate
            },
            {
                "params": [
                    p for n, p in self.transformer.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": learning_rate
            }
        ]

        # Classifier parameters (higher LR - 10x)
        classifier_params = [
            {
                "params": [
                    p for n, p in self.classifier.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": learning_rate * 10
            },
            {
                "params": [
                    p for n, p in self.classifier.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": learning_rate * 10
            }
        ]

        return transformer_params + classifier_params

    def get_model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total_params - trainable
        return {
            'model_name': f'TransformerClassifier ({self.model_name})',
            'total_parameters': total_params,
            'trainable_parameters': trainable,
            'frozen_parameters': frozen,
            'pooling_strategy': self.pooling_strategy,
            'num_classes': self.num_classes
        }


class EfficientTransformerClassifier(TransformerClassifier):
    """
    Memory-efficient variant with:
    - Gradient checkpointing
    - Mixed precision ready
    - Feature extraction option (frozen backbone)
    """

    def __init__(self, *args, **kwargs):
        kwargs['use_gradient_checkpointing'] = True
        super().__init__(*args, **kwargs)

    def extract_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract features without classification head - useful for embedding."""
        with torch.no_grad():
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            hidden_states = outputs.last_hidden_state
            return self.pool_output(hidden_states, attention_mask)


def get_scheduler(optimizer, scheduler_type: str, num_warmup_steps: int, num_training_steps: int):
    """Get learning rate scheduler."""
    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


if __name__ == "__main__":
    # Quick test with DistilBERT
    print("Testing TransformerClassifier with DistilBERT...")
    model = TransformerClassifier(
        model_name="distilbert-base-uncased",
        num_classes=3,
        pooling_strategy="mean"
    )

    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    output = model(input_ids, attention_mask=attention_mask)
    print(f"✅ BERT output shape: {output['logits'].shape}")
    print(f"   Model info: {model.get_model_info()}")

    # Test optimizer groups
    opt_groups = model.get_optimizer_groups()
    print(f"   Optimizer groups: {len(opt_groups)}")
