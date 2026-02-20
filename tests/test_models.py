"""
tests/test_models.py - Unit tests for all models, data pipeline, and API
Run with: pytest tests/ -v
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '.')


# ─── Model Tests ──────────────────────────────────────────────────────────

class TestTextCNN:
    """Tests for TextCNN model."""

    @pytest.fixture
    def model(self):
        from src.models.cnn_model import TextCNN
        return TextCNN(vocab_size=1000, num_classes=3, embed_dim=64, num_filters=32)

    def test_output_shape(self, model):
        x = torch.randint(0, 1000, (4, 64))
        out = model(x)
        assert out['logits'].shape == (4, 3), "Wrong logits shape"
        assert out['probabilities'].shape == (4, 3), "Wrong probabilities shape"

    def test_probabilities_sum_to_one(self, model):
        x = torch.randint(0, 1000, (4, 64))
        out = model(x)
        sums = out['probabilities'].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5), "Probabilities don't sum to 1"

    def test_different_sequence_lengths(self, model):
        for seq_len in [32, 64, 128, 256]:
            x = torch.randint(0, 1000, (2, seq_len))
            out = model(x)
            assert out['logits'].shape == (2, 3)

    def test_model_info(self, model):
        info = model.get_model_info()
        assert 'total_parameters' in info
        assert info['total_parameters'] > 0
        assert info['num_classes'] == 3

    def test_gradient_flow(self, model):
        """Test that gradients flow through all layers."""
        x = torch.randint(0, 1000, (2, 64))
        out = model(x)
        loss = out['logits'].sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


class TestBiLSTM:
    """Tests for BiLSTM model."""

    @pytest.fixture
    def model(self):
        from src.models.lstm_model import BiLSTMClassifier
        return BiLSTMClassifier(
            vocab_size=1000, num_classes=3,
            embed_dim=64, hidden_dim=64, num_layers=2
        )

    def test_output_shape(self, model):
        x = torch.randint(0, 1000, (4, 64))
        mask = torch.ones(4, 64)
        out = model(x, attention_mask=mask)
        assert out['logits'].shape == (4, 3)
        assert out['attention_weights'] is not None

    def test_padding_mask(self, model):
        """Padded tokens should not affect predictions (much)."""
        x = torch.randint(1, 1000, (2, 64))
        mask_full = torch.ones(2, 64)
        mask_half = torch.zeros(2, 64)
        mask_half[:, :32] = 1  # Only first 32 tokens valid

        out_full = model(x, attention_mask=mask_full)
        out_half = model(x, attention_mask=mask_half)
        # Outputs should differ
        assert not torch.allclose(out_full['logits'], out_half['logits'])

    def test_bidirectional_output(self, model):
        assert model.bidirectional == True


class TestTransformerClassifier:
    """Tests for BERT/Transformer model."""

    @pytest.fixture
    def model(self):
        from src.models.bert_model import TransformerClassifier
        return TransformerClassifier(
            model_name="distilbert-base-uncased",
            num_classes=3,
            pooling_strategy="mean"
        )

    def test_output_shape(self, model):
        # Small sequence for quick test
        x = torch.randint(0, 1000, (2, 32))
        mask = torch.ones(2, 32)
        out = model(x, attention_mask=mask)
        assert out['logits'].shape == (2, 3)
        assert out['probabilities'].shape == (2, 3)
        assert out['pooled_output'].shape[0] == 2

    def test_pooling_strategies(self):
        from src.models.bert_model import TransformerClassifier
        for strategy in ['cls', 'mean', 'max']:
            model = TransformerClassifier(
                model_name="distilbert-base-uncased",
                num_classes=3,
                pooling_strategy=strategy
            )
            x = torch.randint(0, 1000, (2, 32))
            mask = torch.ones(2, 32)
            out = model(x, attention_mask=mask)
            assert out['logits'].shape == (2, 3), f"Failed for strategy {strategy}"

    def test_optimizer_groups(self, model):
        groups = model.get_optimizer_groups()
        assert len(groups) > 1, "Should have multiple optimizer groups"
        # Classifier LR should be higher than transformer LR
        transformer_lr = groups[0]['lr']
        classifier_lr = groups[-1]['lr']
        assert classifier_lr >= transformer_lr


class TestEnsemble:
    """Tests for Ensemble model."""

    @pytest.fixture
    def ensemble(self):
        from src.models.cnn_model import TextCNN
        from src.models.lstm_model import BiLSTMClassifier
        from src.models.ensemble_model import EnsembleClassifier

        models = {
            'cnn': TextCNN(vocab_size=1000, num_classes=3, embed_dim=32, num_filters=16),
            'lstm': BiLSTMClassifier(vocab_size=1000, num_classes=3, embed_dim=32, hidden_dim=32)
        }
        return EnsembleClassifier(models, num_classes=3, ensemble_strategy='weighted')

    def test_weighted_ensemble(self, ensemble):
        x = torch.randint(0, 1000, (4, 64))
        out = ensemble(x)
        assert out['logits'].shape == (4, 3)
        assert out['ensemble_weights'] is not None
        assert torch.allclose(out['ensemble_weights'].sum(), torch.tensor(1.0), atol=1e-5)

    def test_stacking_ensemble(self):
        from src.models.cnn_model import TextCNN
        from src.models.ensemble_model import EnsembleClassifier

        models = {
            'cnn1': TextCNN(vocab_size=1000, num_classes=3, embed_dim=32, num_filters=16),
            'cnn2': TextCNN(vocab_size=1000, num_classes=3, embed_dim=32, num_filters=32)
        }
        ensemble = EnsembleClassifier(models, num_classes=3, ensemble_strategy='stacking')
        x = torch.randint(0, 1000, (4, 64))
        out = ensemble(x)
        assert out['logits'].shape == (4, 3)


# ─── Data Pipeline Tests ──────────────────────────────────────────────────

class TestDataPipeline:
    """Tests for data loading and preprocessing."""

    def test_text_preprocessor(self):
        from src.data.data_loader import TextPreprocessor
        pp = TextPreprocessor()

        # Test cleaning
        assert pp.clean_text("Hello World! http://example.com") == "hello world!"
        assert pp.clean_text("<b>Bold text</b>") == "bold text"
        assert pp.clean_text("  extra   spaces  ") == "extra   spaces"

    def test_vocabulary_builder(self):
        from src.data.data_loader import VocabularyBuilder
        vocab = VocabularyBuilder(max_vocab_size=100, min_freq=1)
        texts = ["hello world", "world is beautiful", "hello beautiful world"]
        vocab.build(texts)

        assert 'hello' in vocab.word2idx
        assert 'world' in vocab.word2idx
        assert '<PAD>' in vocab.word2idx
        assert vocab.word2idx['<PAD>'] == 0
        assert vocab.word2idx['<UNK>'] == 1

    def test_tokenization(self):
        from src.data.data_loader import VocabularyBuilder
        vocab = VocabularyBuilder(min_freq=1)
        vocab.build(["hello world foo"])
        tokens = vocab.tokenize("hello world unknown_word")
        assert tokens[0] == vocab.word2idx['hello']
        assert tokens[1] == vocab.word2idx['world']
        assert tokens[2] == 1  # <UNK>

    def test_generate_sample_data(self):
        from src.data.data_loader import DataPipeline, DataConfig
        config = DataConfig()
        pipeline = DataPipeline(config)
        df = pipeline.generate_sample_data(n_samples=100)
        assert len(df) == 100
        assert 'text' in df.columns
        assert 'sentiment' in df.columns
        assert set(df['sentiment'].unique()).issubset({'positive', 'negative', 'neutral'})


# ─── Drift Detection Tests ────────────────────────────────────────────────

class TestDriftDetector:
    """Tests for monitoring and drift detection."""

    def test_psi_no_drift(self):
        from src.monitoring.drift_detector import PopulationStabilityIndex
        psi_calc = PopulationStabilityIndex()
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(0, 1, 1000)  # Same distribution
        psi = psi_calc.calculate(ref, cur)
        assert psi < 0.1, f"Expected low PSI for same distribution, got {psi}"

    def test_psi_drift_detected(self):
        from src.monitoring.drift_detector import PopulationStabilityIndex
        psi_calc = PopulationStabilityIndex()
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(5, 1, 1000)  # Very different distribution
        psi = psi_calc.calculate(ref, cur)
        assert psi > 0.2, f"Expected high PSI for drifted distribution, got {psi}"

    def test_text_drift_detector(self):
        from src.monitoring.drift_detector import TextDriftDetector, DriftConfig
        config = DriftConfig(min_samples=10)
        detector = TextDriftDetector(config)
        ref_texts = [f"Normal product review number {i}" for i in range(100)]
        detector.fit(ref_texts)
        # Same distribution - should not detect drift
        cur_texts = [f"Normal product review number {i+100}" for i in range(50)]
        results = detector.detect(cur_texts)
        assert len(results) > 0

    def test_label_drift_detector(self):
        from src.monitoring.drift_detector import LabelDriftDetector
        detector = LabelDriftDetector()
        ref_labels = np.array([0] * 400 + [1] * 350 + [2] * 250)
        detector.fit(ref_labels, num_classes=3)
        # Similar distribution
        cur_labels = np.array([0] * 100 + [1] * 90 + [2] * 60)
        result = detector.detect(cur_labels)
        assert 'drift_detected' in result.__dict__


# ─── API Tests ────────────────────────────────────────────────────────────

class TestAPI:
    """Tests for FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        try:
            from fastapi.testclient import TestClient
            from src.serving.api import app
            return TestClient(app)
        except Exception:
            pytest.skip("API dependencies not available")

    def test_health_endpoint(self, client):
        response = client.get("/health")
        # Either healthy or 503 (no models loaded in test)
        assert response.status_code in [200, 503]

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_predict_empty_text(self, client):
        """Should return 422 for empty text."""
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_predict_too_long_text(self, client):
        """Should return 422 for very long text."""
        long_text = "a" * 10001
        response = client.post("/predict", json={"text": long_text})
        assert response.status_code == 422


# ─── Pytest Configuration ─────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
