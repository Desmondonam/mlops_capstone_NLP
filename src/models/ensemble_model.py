"""
ensemble_model.py - Ensemble of CNN + BiLSTM + BERT models
Strategies: Average, Weighted Average, Stacking (meta-learner), Voting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EnsembleClassifier(nn.Module):
    """
    Ensemble model that combines multiple base classifiers.
    
    Supports multiple ensemble strategies:
    1. Simple average of probabilities
    2. Weighted average (learned or manual weights)  
    3. Stacking with meta-learner
    4. Majority voting (for inference only)
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        num_classes: int = 3,
        ensemble_strategy: str = "weighted",  # average, weighted, stacking
        model_weights: Optional[List[float]] = None
    ):
        super(EnsembleClassifier, self).__init__()

        self.model_names = list(models.keys())
        self.models = nn.ModuleDict(models)
        self.num_classes = num_classes
        self.ensemble_strategy = ensemble_strategy
        self.num_models = len(models)

        if ensemble_strategy == "weighted":
            if model_weights:
                assert len(model_weights) == self.num_models
                weights = torch.tensor(model_weights, dtype=torch.float32)
            else:
                # Initialize as uniform, will be learned during training
                weights = torch.ones(self.num_models) / self.num_models
            self.model_weights = nn.Parameter(weights)

        elif ensemble_strategy == "stacking":
            # Meta-learner that takes concatenated predictions as input
            self.meta_learner = nn.Sequential(
                nn.Linear(num_classes * self.num_models, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mode: str = "bert",  # which input format to use
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass through all models and ensemble their predictions.
        """
        all_logits = []
        all_probs = []
        model_outputs = {}

        for name, model in self.models.items():
            # Each model may need different inputs
            if hasattr(model, 'transformer'):
                # BERT-style model
                out = model(input_ids, attention_mask=attention_mask)
            else:
                # CNN/LSTM style model
                out = model(input_ids)

            all_logits.append(out['logits'])
            all_probs.append(out['probabilities'])
            model_outputs[name] = out

        # Stack: (num_models, batch, num_classes)
        stacked_probs = torch.stack(all_probs, dim=0)
        stacked_logits = torch.stack(all_logits, dim=0)

        if self.ensemble_strategy == "average":
            final_probs = stacked_probs.mean(dim=0)
            final_logits = stacked_logits.mean(dim=0)

        elif self.ensemble_strategy == "weighted":
            # Normalize weights with softmax
            weights = F.softmax(self.model_weights, dim=0)
            # weights: (num_models,) → (num_models, 1, 1)
            weights = weights.view(-1, 1, 1)
            final_probs = (stacked_probs * weights).sum(dim=0)
            final_logits = (stacked_logits * weights).sum(dim=0)

        elif self.ensemble_strategy == "stacking":
            # Concatenate all model probabilities and pass through meta-learner
            concat_probs = stacked_probs.permute(1, 0, 2).reshape(
                stacked_probs.shape[1], -1
            )  # (batch, num_models * num_classes)
            final_logits = self.meta_learner(concat_probs)
            final_probs = F.softmax(final_logits, dim=-1)

        return {
            'logits': final_logits,
            'probabilities': final_probs,
            'model_outputs': model_outputs,
            'ensemble_weights': F.softmax(self.model_weights, dim=0).detach()
            if self.ensemble_strategy == "weighted" else None
        }

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        """Inference method with voting option."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(*args, **kwargs)

        # Majority voting
        votes = []
        for probs in outputs['model_outputs'].values():
            votes.append(probs['probabilities'].argmax(dim=-1))

        vote_tensor = torch.stack(votes, dim=0)  # (num_models, batch)
        majority_votes = []
        for i in range(vote_tensor.shape[1]):
            counts = torch.bincount(vote_tensor[:, i], minlength=self.num_classes)
            majority_votes.append(counts.argmax())

        outputs['majority_vote'] = torch.stack(majority_votes)
        return outputs


class ModelRouter:
    """
    Production model router for A/B testing and canary deployments.
    Routes requests to different models based on configured split.
    """

    def __init__(
        self,
        models: Dict[str, Any],
        routing_config: Dict[str, float]
    ):
        """
        Args:
            models: dict of model_name -> model
            routing_config: dict of model_name -> traffic_fraction (must sum to 1.0)
        """
        assert abs(sum(routing_config.values()) - 1.0) < 1e-6, "Traffic fractions must sum to 1.0"

        self.models = models
        self.routing_config = routing_config
        self.model_names = list(routing_config.keys())
        self.traffic_fractions = list(routing_config.values())
        self.request_counts = {name: 0 for name in self.model_names}

    def route(self, request_id: Optional[str] = None) -> tuple:
        """
        Route a request to a model.
        Returns: (model_name, model)
        """
        # Hash-based routing for consistent model assignment
        if request_id:
            idx = hash(request_id) % 100 / 100.0
        else:
            idx = np.random.random()

        cumulative = 0.0
        for name, fraction in zip(self.model_names, self.traffic_fractions):
            cumulative += fraction
            if idx <= cumulative:
                self.request_counts[name] += 1
                return name, self.models[name]

        # Fallback to last model
        name = self.model_names[-1]
        self.request_counts[name] += 1
        return name, self.models[name]

    def get_stats(self) -> dict:
        """Get routing statistics."""
        total = sum(self.request_counts.values())
        return {
            name: {
                'count': count,
                'actual_fraction': count / total if total > 0 else 0,
                'target_fraction': self.routing_config[name]
            }
            for name, count in self.request_counts.items()
        }


if __name__ == "__main__":
    import sys
    sys.path.append('../../')
    from src.models.cnn_model import TextCNN
    from src.models.lstm_model import BiLSTMClassifier

    print("Testing EnsembleClassifier...")

    vocab_size, num_classes = 10000, 3
    batch_size, seq_len = 4, 128

    models = {
        "cnn": TextCNN(vocab_size=vocab_size, num_classes=num_classes),
        "lstm": BiLSTMClassifier(vocab_size=vocab_size, num_classes=num_classes)
    }

    ensemble = EnsembleClassifier(
        models=models,
        num_classes=num_classes,
        ensemble_strategy="weighted"
    )

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = ensemble(x)
    print(f"✅ Ensemble output: {output['logits'].shape}")
    print(f"   Learned weights: {output['ensemble_weights']}")

    # Test router
    router = ModelRouter(
        models={"cnn": models["cnn"], "lstm": models["lstm"]},
        routing_config={"cnn": 0.2, "lstm": 0.8}
    )
    for _ in range(10):
        name, model = router.route()
    print(f"✅ Router stats: {router.get_stats()}")
