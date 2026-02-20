"""
drift_detector.py - Data Drift & Model Performance Monitoring

Covers:
- Statistical drift detection (PSI, KS test, Chi-Square)
- Model performance degradation detection
- Automated retraining triggers
- Evidently AI integration
- Alert generation
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
from collections import defaultdict
from datetime import datetime, timedelta
import warnings

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, ClassificationPreset
    from evidently.metrics import (
        DatasetDriftMetric,
        DatasetMissingValuesSummaryMetric,
        ClassificationQualityMetric
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    warnings.warn("Evidently AI not installed. Using basic drift detection only.")

logger = logging.getLogger(__name__)


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    psi_threshold: float = 0.2        # PSI > 0.2 = significant drift
    ks_threshold: float = 0.05        # KS test p-value threshold
    chi2_threshold: float = 0.05      # Chi-square test p-value
    performance_threshold: float = 0.05  # 5% performance drop triggers alert
    min_samples: int = 100            # Minimum samples for drift detection
    window_size: int = 1000           # Rolling window for online detection
    alert_cooldown_hours: int = 24    # Cooldown between alerts


@dataclass
class DriftResult:
    """Result of a drift detection check."""
    feature: str
    drift_type: str  # data_drift, label_drift, performance_drift
    drift_detected: bool
    score: float
    threshold: float
    p_value: Optional[float] = None
    details: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class PopulationStabilityIndex:
    """
    Population Stability Index (PSI) for detecting feature drift.
    PSI < 0.1: No significant change
    PSI 0.1-0.2: Moderate change, worth investigating
    PSI > 0.2: Significant change - model retraining recommended
    """

    @staticmethod
    def calculate(
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Calculate PSI between reference and current distributions."""
        # Create bins based on reference distribution
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)

        if len(bins) < 2:
            return 0.0

        # Calculate proportions
        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        # Normalize
        ref_pct = (ref_counts + 0.0001) / len(reference)
        cur_pct = (cur_counts + 0.0001) / len(current)

        # PSI formula
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)


class TextDriftDetector:
    """
    Detect drift in text data using statistical tests on text features.
    Monitors: text length, vocabulary, sentiment distribution
    """

    def __init__(self, config: DriftConfig = None):
        self.config = config or DriftConfig()
        self.reference_stats = None
        self.psi_calculator = PopulationStabilityIndex()

    def compute_text_features(self, texts: List[str]) -> pd.DataFrame:
        """Extract statistical features from a list of texts."""
        features = []
        for text in texts:
            words = text.split()
            features.append({
                'length_chars': len(text),
                'length_words': len(words),
                'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
                'punct_ratio': sum(1 for c in text if c in '.,!?;:') / max(len(text), 1),
                'upper_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
                'unique_words_ratio': len(set(words)) / max(len(words), 1)
            })
        return pd.DataFrame(features)

    def fit(self, reference_texts: List[str]):
        """Fit on reference (training) data."""
        logger.info(f"Fitting drift detector on {len(reference_texts)} reference samples")
        self.reference_features = self.compute_text_features(reference_texts)
        self.reference_stats = {
            col: {
                'mean': float(self.reference_features[col].mean()),
                'std': float(self.reference_features[col].std()),
                'min': float(self.reference_features[col].min()),
                'max': float(self.reference_features[col].max()),
                'values': self.reference_features[col].values
            }
            for col in self.reference_features.columns
        }
        logger.info("✅ Drift detector fitted")
        return self

    def detect(self, current_texts: List[str]) -> List[DriftResult]:
        """Detect drift in current data vs reference."""
        if self.reference_stats is None:
            raise ValueError("Must call fit() before detect()")

        if len(current_texts) < self.config.min_samples:
            logger.warning(f"Only {len(current_texts)} samples, need {self.config.min_samples} for reliable drift detection")

        current_features = self.compute_text_features(current_texts)
        results = []

        for feature in self.reference_features.columns:
            ref_values = self.reference_stats[feature]['values']
            cur_values = current_features[feature].values

            # PSI calculation
            psi = self.psi_calculator.calculate(ref_values, cur_values)

            # KS test
            ks_stat, ks_pvalue = ks_2samp(ref_values, cur_values)

            # Combine: drift detected if PSI > threshold OR KS test significant
            drift_detected = (
                psi > self.config.psi_threshold or
                ks_pvalue < self.config.ks_threshold
            )

            results.append(DriftResult(
                feature=feature,
                drift_type="data_drift",
                drift_detected=drift_detected,
                score=psi,
                threshold=self.config.psi_threshold,
                p_value=ks_pvalue,
                details={
                    'psi': psi,
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(ks_pvalue),
                    'ref_mean': float(ref_values.mean()),
                    'cur_mean': float(cur_values.mean()),
                    'ref_std': float(ref_values.std()),
                    'cur_std': float(cur_values.std())
                }
            ))

        return results


class LabelDriftDetector:
    """Detect drift in prediction distributions (label drift)."""

    def __init__(self, config: DriftConfig = None):
        self.config = config or DriftConfig()
        self.reference_distribution = None

    def fit(self, reference_labels: np.ndarray, num_classes: int):
        """Fit on reference label distribution."""
        counts = np.bincount(reference_labels, minlength=num_classes)
        self.reference_distribution = counts / counts.sum()
        self.num_classes = num_classes
        logger.info(f"Reference label distribution: {self.reference_distribution}")

    def detect(self, current_labels: np.ndarray) -> DriftResult:
        """Detect drift in current label distribution."""
        if self.reference_distribution is None:
            raise ValueError("Must call fit() first")

        counts = np.bincount(current_labels, minlength=self.num_classes)
        current_distribution = counts / counts.sum()

        # Chi-square test
        expected = self.reference_distribution * len(current_labels)
        observed = counts

        # Add small epsilon to avoid zero issues
        expected = np.maximum(expected, 1e-10)

        chi2_stat, chi2_pvalue = stats.chisquare(observed, expected)

        # Jensen-Shannon divergence
        p = self.reference_distribution + 1e-10
        q = current_distribution + 1e-10
        m = 0.5 * (p + q)
        js_div = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

        drift_detected = chi2_pvalue < self.config.chi2_threshold

        return DriftResult(
            feature="label_distribution",
            drift_type="label_drift",
            drift_detected=drift_detected,
            score=float(js_div),
            threshold=0.1,  # JSD threshold
            p_value=float(chi2_pvalue),
            details={
                'reference_distribution': self.reference_distribution.tolist(),
                'current_distribution': current_distribution.tolist(),
                'chi2_statistic': float(chi2_stat),
                'js_divergence': float(js_div)
            }
        )


class PerformanceMonitor:
    """
    Monitor model performance over time and detect degradation.
    Uses sliding window approach.
    """

    def __init__(self, window_size: int = 100, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.baseline_metrics = None
        self.recent_predictions = []
        self.recent_labels = []

    def set_baseline(self, metrics: Dict[str, float]):
        """Set baseline performance metrics from validation set."""
        self.baseline_metrics = metrics
        logger.info(f"Baseline metrics set: {metrics}")

    def update(self, predictions: List[int], true_labels: List[int]):
        """Update with new predictions and labels."""
        self.recent_predictions.extend(predictions)
        self.recent_labels.extend(true_labels)

        # Keep only window_size most recent
        if len(self.recent_predictions) > self.window_size:
            self.recent_predictions = self.recent_predictions[-self.window_size:]
            self.recent_labels = self.recent_labels[-self.window_size:]

    def check_degradation(self) -> Optional[DriftResult]:
        """Check if model performance has degraded."""
        if self.baseline_metrics is None:
            return None
        if len(self.recent_predictions) < 50:
            return None

        from sklearn.metrics import accuracy_score, f1_score
        current_accuracy = accuracy_score(self.recent_labels, self.recent_predictions)
        current_f1 = f1_score(self.recent_labels, self.recent_predictions,
                               average='macro', zero_division=0)

        baseline_acc = self.baseline_metrics.get('accuracy', 0)
        acc_drop = baseline_acc - current_accuracy

        degraded = acc_drop > self.threshold

        return DriftResult(
            feature="model_performance",
            drift_type="performance_drift",
            drift_detected=degraded,
            score=acc_drop,
            threshold=self.threshold,
            details={
                'baseline_accuracy': baseline_acc,
                'current_accuracy': current_accuracy,
                'accuracy_drop': acc_drop,
                'current_f1': current_f1,
                'sample_count': len(self.recent_predictions)
            }
        )


class DriftMonitoringPipeline:
    """
    Complete drift monitoring pipeline that combines all detectors.
    Can trigger automated retraining on drift detection.
    """

    def __init__(
        self,
        config: DriftConfig = None,
        retraining_callback: Optional[Any] = None
    ):
        self.config = config or DriftConfig()
        self.text_detector = TextDriftDetector(config)
        self.label_detector = LabelDriftDetector(config)
        self.performance_monitor = PerformanceMonitor(
            window_size=config.window_size if config else 1000,
            threshold=config.performance_threshold if config else 0.05
        )
        self.retraining_callback = retraining_callback
        self.alerts = []
        self.last_alert_time = None
        self.drift_history = []

    def fit(
        self,
        reference_texts: List[str],
        reference_labels: np.ndarray,
        baseline_metrics: Dict[str, float],
        num_classes: int
    ):
        """Fit all detectors on reference data."""
        self.text_detector.fit(reference_texts)
        self.label_detector.fit(reference_labels, num_classes)
        self.performance_monitor.set_baseline(baseline_metrics)
        logger.info("✅ All drift detectors fitted")
        return self

    def monitor(
        self,
        current_texts: List[str],
        predictions: Optional[List[int]] = None,
        true_labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Run all drift checks on new data.
        Returns comprehensive drift report.
        """
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'sample_count': len(current_texts),
            'data_drift': [],
            'label_drift': None,
            'performance_drift': None,
            'overall_drift_detected': False,
            'alert_triggered': False,
            'recommendations': []
        }

        # 1. Check data (text feature) drift
        text_drift_results = self.text_detector.detect(current_texts)
        results['data_drift'] = [
            {
                'feature': r.feature,
                'drift_detected': r.drift_detected,
                'psi': r.score,
                'p_value': r.p_value
            }
            for r in text_drift_results
        ]

        # 2. Check label/prediction drift
        if predictions is not None:
            label_result = self.label_detector.detect(np.array(predictions))
            results['label_drift'] = {
                'drift_detected': label_result.drift_detected,
                'js_divergence': label_result.score,
                'details': label_result.details
            }

        # 3. Check performance drift
        if predictions is not None and true_labels is not None:
            self.performance_monitor.update(predictions, true_labels)
            perf_result = self.performance_monitor.check_degradation()
            if perf_result:
                results['performance_drift'] = {
                    'drift_detected': perf_result.drift_detected,
                    'accuracy_drop': perf_result.score,
                    'details': perf_result.details
                }

        # 4. Determine overall drift
        any_data_drift = any(r.drift_detected for r in text_drift_results)
        label_drift = results['label_drift'] and results['label_drift']['drift_detected']
        perf_drift = results['performance_drift'] and results['performance_drift']['drift_detected']

        results['overall_drift_detected'] = any_data_drift or label_drift or perf_drift

        # 5. Generate recommendations
        if any_data_drift:
            results['recommendations'].append("⚠️ Data distribution shift detected. Consider data collection review.")
        if label_drift:
            results['recommendations'].append("⚠️ Prediction distribution shifted. Check for data quality issues.")
        if perf_drift:
            results['recommendations'].append("🚨 Performance degradation detected. Retraining recommended!")

        # 6. Trigger alert if needed
        if results['overall_drift_detected']:
            should_alert = (
                self.last_alert_time is None or
                (datetime.utcnow() - self.last_alert_time).total_seconds() >
                self.config.alert_cooldown_hours * 3600
            )
            if should_alert:
                self._trigger_alert(results)
                results['alert_triggered'] = True

        # Store in history
        self.drift_history.append({
            'timestamp': results['timestamp'],
            'drift_detected': results['overall_drift_detected']
        })

        return results

    def _trigger_alert(self, drift_report: dict):
        """Trigger drift alert and optionally initiate retraining."""
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'drift_detected': True,
            'details': drift_report['recommendations']
        }
        self.alerts.append(alert)
        self.last_alert_time = datetime.utcnow()

        logger.warning(f"🚨 DRIFT ALERT: {drift_report['recommendations']}")

        # Trigger retraining if callback provided
        if self.retraining_callback:
            logger.info("Triggering automated retraining pipeline...")
            self.retraining_callback(drift_report)

    def generate_evidently_report(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        output_path: str = "drift_report.html"
    ):
        """Generate rich HTML drift report using Evidently AI."""
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available, skipping HTML report")
            return

        report = Report(metrics=[
            DataDriftPreset(),
            DatasetMissingValuesSummaryMetric()
        ])
        report.run(reference_data=reference_df, current_data=current_df)
        report.save_html(output_path)
        logger.info(f"✅ Evidently report saved to {output_path}")

    def get_drift_summary(self) -> dict:
        """Get summary of drift detection history."""
        if not self.drift_history:
            return {"message": "No drift history available"}

        total = len(self.drift_history)
        drifts = sum(1 for h in self.drift_history if h['drift_detected'])
        return {
            'total_checks': total,
            'drift_detected_count': drifts,
            'drift_rate': drifts / total if total > 0 else 0,
            'total_alerts': len(self.alerts),
            'last_check': self.drift_history[-1]['timestamp']
        }


if __name__ == "__main__":
    import numpy as np

    print("Testing DriftMonitoringPipeline...")

    # Create synthetic data
    np.random.seed(42)
    reference_texts = [
        f"This is a {'great' if i % 2 == 0 else 'bad'} product review number {i}"
        for i in range(500)
    ]
    reference_labels = np.array([0, 1, 2] * 167)[:500]
    baseline_metrics = {'accuracy': 0.85, 'f1_macro': 0.84}

    # Initialize pipeline
    pipeline = DriftMonitoringPipeline()
    pipeline.fit(reference_texts, reference_labels, baseline_metrics, num_classes=3)

    # Simulate drifted data (different distribution)
    drifted_texts = [
        f"Product review with very different text pattern: {i * 100}"
        for i in range(200)
    ]
    drifted_predictions = np.random.choice([0, 1, 2], size=200, p=[0.6, 0.3, 0.1])

    # Monitor
    report = pipeline.monitor(drifted_texts, predictions=drifted_predictions.tolist())
    print(f"✅ Drift report generated:")
    print(f"   Overall drift: {report['overall_drift_detected']}")
    print(f"   Recommendations: {report['recommendations']}")
    print(f"   Summary: {pipeline.get_drift_summary()}")
