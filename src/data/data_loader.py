"""
data_loader.py - Data Ingestion, Preprocessing & Dataset Creation
Covers: Data pipelines, tokenization, dataset splitting, augmentation
"""

import os
import re
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_path: str = "data/raw/reviews.csv"
    max_length: int = 256
    test_size: float = 0.2
    val_size: float = 0.1
    batch_size: int = 32
    num_workers: int = 4
    text_column: str = "text"
    label_column: str = "sentiment"
    tokenizer_name: str = "distilbert-base-uncased"
    augment_data: bool = True
    random_seed: int = 42


class TextPreprocessor:
    """
    Text cleaning and preprocessing pipeline.
    Handles: noise removal, normalization, lemmatization
    """

    def __init__(self, remove_stopwords: bool = False):
        self.remove_stopwords = remove_stopwords
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()

    def clean_text(self, text: str) -> str:
        """Full text cleaning pipeline."""
        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Optionally remove stopwords
        if self.remove_stopwords:
            words = text.split()
            words = [w for w in words if w not in self.stop_words]
            text = ' '.join(words)

        return text

    def augment_text(self, text: str) -> List[str]:
        """
        Simple data augmentation techniques:
        - Synonym replacement (simplified)
        - Random word deletion
        - Random word swap
        """
        augmented = [text]

        # Random word deletion (10% of words)
        words = text.split()
        if len(words) > 5:
            n_delete = max(1, int(len(words) * 0.1))
            indices_to_delete = np.random.choice(len(words), n_delete, replace=False)
            deleted_words = [w for i, w in enumerate(words) if i not in indices_to_delete]
            augmented.append(' '.join(deleted_words))

        # Random word swap
        if len(words) > 3:
            swapped = words.copy()
            idx1, idx2 = np.random.choice(len(words), 2, replace=False)
            swapped[idx1], swapped[idx2] = swapped[idx2], swapped[idx1]
            augmented.append(' '.join(swapped))

        return augmented


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment classification.
    Compatible with both traditional models (vocab-based) and transformers.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 256,
        mode: str = "bert"  # "bert" or "classical"
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        if self.mode == "bert":
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            # Classical mode: return token indices
            tokens = self.tokenizer(text)
            # Pad or truncate
            if len(tokens) < self.max_length:
                tokens = tokens + [0] * (self.max_length - len(tokens))
            else:
                tokens = tokens[:self.max_length]

            return {
                'input_ids': torch.tensor(tokens, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)
            }


class VocabularyBuilder:
    """Build vocabulary for classical DL models (CNN, LSTM)."""

    def __init__(self, max_vocab_size: int = 30000, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = {}

    def build(self, texts: List[str]):
        """Build vocabulary from training texts."""
        logger.info("Building vocabulary...")

        # Count word frequencies
        for text in texts:
            for word in text.lower().split():
                self.word_freq[word] = self.word_freq.get(word, 0) + 1

        # Sort by frequency and filter by min_freq
        sorted_words = sorted(
            self.word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Build vocab (leave space for PAD and UNK)
        vocab_words = [
            word for word, freq in sorted_words
            if freq >= self.min_freq
        ][:self.max_vocab_size - 2]

        for idx, word in enumerate(vocab_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        logger.info(f"Vocabulary size: {len(self.word2idx)}")
        return self

    def tokenize(self, text: str) -> List[int]:
        """Convert text to token indices."""
        return [
            self.word2idx.get(word, 1)  # 1 = UNK
            for word in text.lower().split()
        ]

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.word2idx, f)

    @classmethod
    def load(cls, path: str):
        vocab = cls()
        with open(path, 'r') as f:
            vocab.word2idx = json.load(f)
        vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
        return vocab


class DataPipeline:
    """
    Full data pipeline: load → validate → preprocess → split → create loaders
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.preprocessor = TextPreprocessor()
        self.label_encoder = LabelEncoder()
        self.vocab = VocabularyBuilder()

        # Load BERT tokenizer
        logger.info(f"Loading tokenizer: {config.tokenizer_name}")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def generate_sample_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate synthetic training data for demo purposes.
        In production, replace with real data loading.
        """
        logger.info(f"Generating {n_samples} synthetic samples...")

        positive_templates = [
            "This product is absolutely {adj}! I {verb} it.",
            "Amazing {noun}. Best purchase I've ever made.",
            "Highly recommend this {noun}. {adj} quality.",
            "Outstanding {adj} experience. Will buy again!",
            "The {noun} exceeded all my expectations. {adj}!",
        ]
        negative_templates = [
            "This {noun} is terrible. Very {adj_neg} experience.",
            "Disappointed with the {noun}. Would not recommend.",
            "Waste of money. The {noun} broke after {time}.",
            "Horrible {adj_neg} service. Never buying again.",
            "The {noun} didn't work as advertised. Very {adj_neg}.",
        ]
        neutral_templates = [
            "The {noun} is okay. Nothing special about it.",
            "Average {noun}. Gets the job done I suppose.",
            "Neither good nor bad. The {noun} is just average.",
            "It's a decent {noun} for the price.",
            "The {noun} works as expected. Nothing more.",
        ]

        pos_words = {'adj': ['fantastic', 'excellent', 'wonderful', 'great', 'superb'],
                     'verb': ['love', 'adore', 'enjoy', 'cherish'],
                     'noun': ['product', 'item', 'service', 'device', 'tool']}
        neg_words = {'adj_neg': ['awful', 'terrible', 'horrible', 'dreadful', 'poor'],
                     'noun': ['product', 'item', 'service', 'device', 'purchase'],
                     'time': ['one day', 'a week', 'two days', 'immediately']}
        neu_words = {'noun': ['product', 'item', 'device', 'service', 'tool']}

        texts, labels = [], []
        np.random.seed(self.config.random_seed)

        for i in range(n_samples):
            sentiment = np.random.choice(['positive', 'negative', 'neutral'],
                                         p=[0.45, 0.35, 0.20])
            if sentiment == 'positive':
                template = np.random.choice(positive_templates)
                for key, values in pos_words.items():
                    template = template.replace(f'{{{key}}}', np.random.choice(values))
            elif sentiment == 'negative':
                template = np.random.choice(negative_templates)
                for key, values in neg_words.items():
                    template = template.replace(f'{{{key}}}', np.random.choice(values))
            else:
                template = np.random.choice(neutral_templates)
                for key, values in neu_words.items():
                    template = template.replace(f'{{{key}}}', np.random.choice(values))

            texts.append(template)
            labels.append(sentiment)

        df = pd.DataFrame({'text': texts, 'sentiment': labels})
        logger.info(f"Generated data distribution:\n{df['sentiment'].value_counts()}")
        return df

    def load_data(self) -> pd.DataFrame:
        """Load data from file or generate sample data."""
        if os.path.exists(self.config.data_path):
            logger.info(f"Loading data from {self.config.data_path}")
            df = pd.read_csv(self.config.data_path)
        else:
            logger.warning(f"Data file not found. Generating sample data...")
            df = self.generate_sample_data()

        # Basic validation
        assert self.config.text_column in df.columns, f"Column '{self.config.text_column}' not found"
        assert self.config.label_column in df.columns, f"Column '{self.config.label_column}' not found"

        # Drop nulls
        df = df.dropna(subset=[self.config.text_column, self.config.label_column])
        logger.info(f"Loaded {len(df)} samples")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply text preprocessing."""
        logger.info("Preprocessing texts...")
        df = df.copy()
        df['text_clean'] = df[self.config.text_column].apply(
            self.preprocessor.clean_text
        )

        # Data augmentation on training data
        if self.config.augment_data:
            logger.info("Applying data augmentation...")
            augmented_rows = []
            for _, row in df.iterrows():
                augmented_texts = self.preprocessor.augment_text(row['text_clean'])
                for aug_text in augmented_texts[1:]:  # Skip original
                    new_row = row.copy()
                    new_row['text_clean'] = aug_text
                    augmented_rows.append(new_row)

            if augmented_rows:
                aug_df = pd.DataFrame(augmented_rows)
                df = pd.concat([df, aug_df], ignore_index=True)
                logger.info(f"After augmentation: {len(df)} samples")

        return df

    def prepare_loaders(
        self,
        mode: str = "bert"
    ) -> Dict[str, DataLoader]:
        """
        Full pipeline: load → preprocess → split → create DataLoaders
        Returns dict with 'train', 'val', 'test' DataLoaders
        """
        # Load and preprocess
        df = self.load_data()
        df = self.preprocess(df)

        # Encode labels
        labels = self.label_encoder.fit_transform(df[self.config.label_column])
        n_classes = len(self.label_encoder.classes_)
        logger.info(f"Classes: {self.label_encoder.classes_}")

        texts = df['text_clean'].tolist()

        # Train/val/test split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.val_size,
            random_state=self.config.random_seed,
            stratify=y_train
        )

        logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Choose tokenizer
        if mode == "bert":
            tokenizer = self.bert_tokenizer
        else:
            # Build vocab for classical models
            self.vocab.build(X_train)
            tokenizer = self.vocab.tokenize

        # Create datasets
        train_dataset = SentimentDataset(X_train, y_train, tokenizer,
                                          self.config.max_length, mode)
        val_dataset = SentimentDataset(X_val, y_val, tokenizer,
                                        self.config.max_length, mode)
        test_dataset = SentimentDataset(X_test, y_test, tokenizer,
                                         self.config.max_length, mode)

        # Create DataLoaders
        loaders = {
            'train': DataLoader(train_dataset, batch_size=self.config.batch_size,
                                shuffle=True, num_workers=0, pin_memory=True),
            'val': DataLoader(val_dataset, batch_size=self.config.batch_size,
                              shuffle=False, num_workers=0),
            'test': DataLoader(test_dataset, batch_size=self.config.batch_size,
                               shuffle=False, num_workers=0)
        }

        return loaders, n_classes


if __name__ == "__main__":
    config = DataConfig()
    pipeline = DataPipeline(config)
    loaders, n_classes = pipeline.prepare_loaders(mode="bert")
    print(f"✅ Data pipeline ready! Classes: {n_classes}")
    print(f"   Train batches: {len(loaders['train'])}")
    print(f"   Val batches: {len(loaders['val'])}")
    print(f"   Test batches: {len(loaders['test'])}")
