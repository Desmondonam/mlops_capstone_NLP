#!/bin/bash

# ─── Run Full ML Pipeline ─────────────────────────────────────────────────
echo "🚀 Starting ML Training Pipeline..."
export PYTHONPATH=$(pwd)
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-"http://localhost:5000"}

echo "📊 Training CNN model..."
python -c "
from src.data.data_loader import DataPipeline, DataConfig
from src.models.cnn_model import TextCNN
from src.training.trainer import Trainer, TrainingConfig

config = DataConfig(batch_size=32)
pipeline = DataPipeline(config)
loaders, n_classes = pipeline.prepare_loaders(mode='classical')
vocab_size = len(pipeline.vocab.word2idx)

model = TextCNN(vocab_size=vocab_size, num_classes=n_classes)
train_config = TrainingConfig(model_type='cnn', num_classes=n_classes, epochs=5)
trainer = Trainer(model, train_config)
results = trainer.train(loaders['train'], loaders['val'], loaders['test'])
print(f'CNN Complete: F1={results[\"best_val_score\"]:.4f}')
"

echo "📊 Training BERT model..."
python -c "
from src.data.data_loader import DataPipeline, DataConfig
from src.models.bert_model import TransformerClassifier
from src.training.trainer import Trainer, TrainingConfig

config = DataConfig(batch_size=16)
pipeline = DataPipeline(config)
loaders, n_classes = pipeline.prepare_loaders(mode='bert')

model = TransformerClassifier(model_name='distilbert-base-uncased', num_classes=n_classes)
train_config = TrainingConfig(model_type='bert', num_classes=n_classes, epochs=5, learning_rate=2e-5)
trainer = Trainer(model, train_config)
results = trainer.train(loaders['train'], loaders['val'], loaders['test'])
print(f'BERT Complete: F1={results[\"best_val_score\"]:.4f}')
"

echo "✅ Pipeline complete! Visit http://localhost:5000 for MLflow results"
echo "🌐 API running at http://localhost:8000"
