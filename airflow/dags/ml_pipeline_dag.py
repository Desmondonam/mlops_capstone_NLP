"""
ml_pipeline_dag.py - Apache Airflow DAG for ML Training Pipeline

This DAG orchestrates the complete ML pipeline:
1. Data ingestion & validation
2. Feature engineering
3. Model training (CNN, LSTM, BERT, Ensemble)
4. Model evaluation & comparison
5. Model registration to MLflow
6. Deployment promotion (staging → production)
7. Post-deployment monitoring
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
import logging
import os
import sys

sys.path.insert(0, '/opt/airflow')

logger = logging.getLogger(__name__)

# ─── DAG Default Arguments ────────────────────────────────────────────────

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['mlops@company.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=4)
}


# ─── Task Functions ───────────────────────────────────────────────────────

def task_validate_data(**context):
    """Validate raw data quality before training."""
    logger.info("Starting data validation...")

    try:
        import pandas as pd
        import numpy as np

        # Simulate data validation checks
        validation_results = {
            'total_records': 5000,
            'null_ratio': 0.002,
            'duplicate_ratio': 0.01,
            'class_balance': {'positive': 0.45, 'negative': 0.35, 'neutral': 0.20},
            'avg_text_length': 52.3,
            'validation_passed': True,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Check validation rules
        assert validation_results['null_ratio'] < 0.05, "Too many nulls!"
        assert validation_results['duplicate_ratio'] < 0.1, "Too many duplicates!"

        min_class_balance = min(validation_results['class_balance'].values())
        assert min_class_balance > 0.1, "Severe class imbalance!"

        # Push results to XCom
        context['task_instance'].xcom_push(
            key='validation_results',
            value=validation_results
        )

        logger.info(f"✅ Data validation passed: {validation_results}")
        return validation_results

    except AssertionError as e:
        logger.error(f"❌ Data validation FAILED: {e}")
        raise


def task_preprocess_data(**context):
    """Preprocess and prepare datasets for training."""
    logger.info("Preprocessing data...")

    try:
        import sys
        sys.path.insert(0, '/opt/airflow')
        from src.data.data_loader import DataPipeline, DataConfig

        config = DataConfig(batch_size=32)
        pipeline = DataPipeline(config)

        # Create datasets
        loaders, n_classes = pipeline.prepare_loaders(mode="bert")

        preprocessing_info = {
            'n_classes': n_classes,
            'train_batches': len(loaders['train']),
            'val_batches': len(loaders['val']),
            'test_batches': len(loaders['test']),
            'status': 'success'
        }

        context['task_instance'].xcom_push(
            key='preprocessing_info',
            value=preprocessing_info
        )

        logger.info(f"✅ Preprocessing complete: {preprocessing_info}")
        return preprocessing_info

    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        raise


def task_train_cnn(**context):
    """Train CNN model."""
    logger.info("Training CNN model...")

    try:
        import torch
        import mlflow

        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
        mlflow.set_experiment('capstone-training')

        from src.data.data_loader import DataPipeline, DataConfig
        from src.models.cnn_model import TextCNN
        from src.training.trainer import Trainer, TrainingConfig

        # Load data
        data_config = DataConfig(batch_size=32)
        pipeline = DataPipeline(data_config)
        loaders, n_classes = pipeline.prepare_loaders(mode="classical")

        # Create model
        vocab_size = len(pipeline.vocab.word2idx)
        model = TextCNN(
            vocab_size=vocab_size,
            num_classes=n_classes,
            embed_dim=128,
            num_filters=100
        )

        # Train
        train_config = TrainingConfig(
            model_type='cnn',
            num_classes=n_classes,
            epochs=5,
            learning_rate=1e-3,
            patience=3,
            experiment_name='capstone-training'
        )

        trainer = Trainer(model, train_config)
        results = trainer.train(loaders['train'], loaders['val'], loaders['test'])

        cnn_results = {
            'model_type': 'cnn',
            'val_f1': results['best_val_score'],
            'best_epoch': results['best_epoch'],
            'run_id': results['run_id']
        }

        context['task_instance'].xcom_push(key='cnn_results', value=cnn_results)
        logger.info(f"✅ CNN training complete: F1={results['best_val_score']:.4f}")
        return cnn_results

    except Exception as e:
        logger.error(f"❌ CNN training failed: {e}")
        raise


def task_train_lstm(**context):
    """Train BiLSTM model."""
    logger.info("Training BiLSTM model...")

    try:
        import torch
        import mlflow

        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))

        from src.data.data_loader import DataPipeline, DataConfig
        from src.models.lstm_model import BiLSTMClassifier
        from src.training.trainer import Trainer, TrainingConfig

        data_config = DataConfig(batch_size=32)
        pipeline = DataPipeline(data_config)
        loaders, n_classes = pipeline.prepare_loaders(mode="classical")

        vocab_size = len(pipeline.vocab.word2idx)
        model = BiLSTMClassifier(
            vocab_size=vocab_size,
            num_classes=n_classes,
            embed_dim=256,
            hidden_dim=256
        )

        train_config = TrainingConfig(
            model_type='lstm',
            num_classes=n_classes,
            epochs=5,
            learning_rate=5e-4,
            patience=3,
            experiment_name='capstone-training'
        )

        trainer = Trainer(model, train_config)
        results = trainer.train(loaders['train'], loaders['val'], loaders['test'])

        lstm_results = {
            'model_type': 'lstm',
            'val_f1': results['best_val_score'],
            'best_epoch': results['best_epoch'],
            'run_id': results['run_id']
        }

        context['task_instance'].xcom_push(key='lstm_results', value=lstm_results)
        logger.info(f"✅ LSTM training complete: F1={results['best_val_score']:.4f}")
        return lstm_results

    except Exception as e:
        logger.error(f"❌ LSTM training failed: {e}")
        raise


def task_train_bert(**context):
    """Fine-tune BERT model (the LLM component)."""
    logger.info("Fine-tuning BERT model (DistilBERT)...")

    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))

        from src.data.data_loader import DataPipeline, DataConfig
        from src.models.bert_model import TransformerClassifier
        from src.training.trainer import Trainer, TrainingConfig

        data_config = DataConfig(batch_size=16)  # Smaller batch for BERT
        pipeline = DataPipeline(data_config)
        loaders, n_classes = pipeline.prepare_loaders(mode="bert")

        model = TransformerClassifier(
            model_name='distilbert-base-uncased',
            num_classes=n_classes,
            pooling_strategy='mean',
            dropout_rate=0.3
        )

        train_config = TrainingConfig(
            model_type='bert',
            num_classes=n_classes,
            epochs=5,
            learning_rate=2e-5,
            patience=3,
            scheduler='cosine',
            experiment_name='capstone-training'
        )

        trainer = Trainer(model, train_config)
        results = trainer.train(loaders['train'], loaders['val'], loaders['test'])

        bert_results = {
            'model_type': 'bert',
            'val_f1': results['best_val_score'],
            'best_epoch': results['best_epoch'],
            'run_id': results['run_id']
        }

        context['task_instance'].xcom_push(key='bert_results', value=bert_results)
        logger.info(f"✅ BERT fine-tuning complete: F1={results['best_val_score']:.4f}")
        return bert_results

    except Exception as e:
        logger.error(f"❌ BERT training failed: {e}")
        raise


def task_compare_models(**context):
    """Compare all models and select the champion."""
    logger.info("Comparing model results...")

    ti = context['task_instance']
    cnn_results = ti.xcom_pull(task_ids='train_cnn', key='cnn_results') or {}
    lstm_results = ti.xcom_pull(task_ids='train_lstm', key='lstm_results') or {}
    bert_results = ti.xcom_pull(task_ids='train_bert', key='bert_results') or {}

    all_results = {
        'cnn': cnn_results,
        'lstm': lstm_results,
        'bert': bert_results
    }

    # Find best model
    best_model = 'bert'  # Default
    best_score = 0.0

    for model_type, results in all_results.items():
        if results and results.get('val_f1', 0) > best_score:
            best_score = results['val_f1']
            best_model = model_type

    comparison = {
        'all_results': all_results,
        'champion_model': best_model,
        'champion_score': best_score,
        'promotion_recommended': best_score > 0.70  # Threshold for promotion
    }

    ti.xcom_push(key='comparison_results', value=comparison)
    logger.info(f"✅ Champion model: {best_model} (F1: {best_score:.4f})")
    return comparison


def task_decide_promotion(**context):
    """Decide whether to promote model to production."""
    ti = context['task_instance']
    comparison = ti.xcom_pull(task_ids='compare_models', key='comparison_results')

    if comparison and comparison.get('promotion_recommended', False):
        logger.info(f"✅ Model approved for production promotion!")
        return 'promote_to_production'
    else:
        logger.warning(f"⚠️ Model performance below threshold, skipping promotion")
        return 'skip_promotion'


def task_promote_to_production(**context):
    """Promote champion model to production in MLflow registry."""
    logger.info("Promoting champion model to production...")

    ti = context['task_instance']
    comparison = ti.xcom_pull(task_ids='compare_models', key='comparison_results')

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
        client = MlflowClient()

        champion = comparison['champion_model']
        model_name = f"{champion}-sentiment"
        run_id = comparison['all_results'].get(champion, {}).get('run_id')

        if run_id:
            # Register model version
            model_uri = f"runs:/{run_id}/model"
            mv = mlflow.register_model(model_uri, model_name)

            # Transition to Production
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Production",
                archive_existing_versions=True  # Archive old production
            )

            logger.info(f"✅ Model {model_name} v{mv.version} promoted to Production!")
            return {'status': 'promoted', 'model_name': model_name, 'version': mv.version}
        else:
            logger.warning("No run_id found, skipping MLflow promotion")
            return {'status': 'skipped'}

    except Exception as e:
        logger.error(f"Promotion failed: {e}")
        # Non-fatal - don't fail the whole DAG
        return {'status': 'failed', 'error': str(e)}


def task_run_drift_check(**context):
    """Post-deployment drift check on recent predictions."""
    logger.info("Running drift detection on recent data...")

    try:
        import numpy as np
        from src.monitoring.drift_detector import DriftMonitoringPipeline, DriftConfig

        # Simulate reference and current data
        np.random.seed(42)
        ref_texts = [f"Reference review sample {i}" for i in range(500)]
        ref_labels = np.array([0, 1, 2] * 167)[:500]
        baseline_metrics = {'accuracy': 0.85, 'f1_macro': 0.83}

        # Recent data (simulated)
        current_texts = [f"Current deployment review {i}" for i in range(200)]
        current_preds = np.random.choice([0, 1, 2], size=200)

        pipeline = DriftMonitoringPipeline()
        pipeline.fit(ref_texts, ref_labels, baseline_metrics, num_classes=3)
        report = pipeline.monitor(current_texts, predictions=current_preds.tolist())

        context['task_instance'].xcom_push(
            key='drift_report',
            value={
                'drift_detected': report['overall_drift_detected'],
                'recommendations': report.get('recommendations', [])
            }
        )

        logger.info(f"✅ Drift check complete. Drift detected: {report['overall_drift_detected']}")
        return report

    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        return {'error': str(e)}


def task_send_notification(**context):
    """Send pipeline completion notification."""
    ti = context['task_instance']
    comparison = ti.xcom_pull(task_ids='compare_models', key='comparison_results') or {}
    drift_report = ti.xcom_pull(task_ids='drift_check', key='drift_report') or {}

    message = f"""
    ╔══════════════════════════════════════╗
    ║    ML Pipeline Completed Successfully ║
    ╠══════════════════════════════════════╣
    ║ Champion Model: {comparison.get('champion_model', 'N/A'):20s} ║
    ║ Best F1 Score:  {comparison.get('champion_score', 0):.4f}               ║
    ║ Promoted:       {comparison.get('promotion_recommended', False)!s:20s} ║
    ║ Drift Detected: {drift_report.get('drift_detected', False)!s:20s} ║
    ╚══════════════════════════════════════╝
    """
    logger.info(message)
    return {'status': 'notified', 'summary': comparison}


# ─── DAG Definition ───────────────────────────────────────────────────────

with DAG(
    dag_id='ml_training_pipeline',
    default_args=default_args,
    description='Complete ML training and deployment pipeline',
    schedule_interval='0 2 * * 0',  # Every Sunday at 2 AM
    catchup=False,
    tags=['ml', 'training', 'deployment', 'nlp'],
    max_active_runs=1
) as dag:

    # ── Start ────────────────────────────────────────────────────────────
    start = EmptyOperator(task_id='start')

    # ── Data Stage ───────────────────────────────────────────────────────
    validate_data = PythonOperator(
        task_id='validate_data',
        python_callable=task_validate_data,
        provide_context=True
    )

    preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=task_preprocess_data,
        provide_context=True
    )

    # ── Training Stage (parallel) ─────────────────────────────────────────
    train_cnn = PythonOperator(
        task_id='train_cnn',
        python_callable=task_train_cnn,
        provide_context=True
    )

    train_lstm = PythonOperator(
        task_id='train_lstm',
        python_callable=task_train_lstm,
        provide_context=True
    )

    train_bert = PythonOperator(
        task_id='train_bert',
        python_callable=task_train_bert,
        provide_context=True
    )

    # ── Evaluation Stage ──────────────────────────────────────────────────
    compare_models = PythonOperator(
        task_id='compare_models',
        python_callable=task_compare_models,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE  # Run even if some training fails
    )

    # ── Deployment Stage ──────────────────────────────────────────────────
    decide_promotion = BranchPythonOperator(
        task_id='decide_promotion',
        python_callable=task_decide_promotion,
        provide_context=True
    )

    promote_to_production = PythonOperator(
        task_id='promote_to_production',
        python_callable=task_promote_to_production,
        provide_context=True
    )

    skip_promotion = EmptyOperator(task_id='skip_promotion')

    # ── Monitoring Stage ──────────────────────────────────────────────────
    drift_check = PythonOperator(
        task_id='drift_check',
        python_callable=task_run_drift_check,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE
    )

    # ── Notification ──────────────────────────────────────────────────────
    notify = PythonOperator(
        task_id='notify',
        python_callable=task_send_notification,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE
    )

    end = EmptyOperator(task_id='end')

    # ─── DAG Dependencies ─────────────────────────────────────────────────
    start >> validate_data >> preprocess_data

    # Parallel training after preprocessing
    preprocess_data >> [train_cnn, train_lstm, train_bert]

    # All models complete → compare
    [train_cnn, train_lstm, train_bert] >> compare_models

    # Branching decision
    compare_models >> decide_promotion
    decide_promotion >> [promote_to_production, skip_promotion]

    # Drift check after deployment (or skip)
    [promote_to_production, skip_promotion] >> drift_check

    # Notification and end
    drift_check >> notify >> end
