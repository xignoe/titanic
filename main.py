#!/usr/bin/env python3
"""
Titanic Survival Predictor - Main Entry Point

This script orchestrates the complete machine learning pipeline for predicting
passenger survival on the Titanic. It supports different execution modes for
training, prediction, and evaluation.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config import ensure_directories, MODEL_CONFIG, OUTPUT_CONFIG


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('titanic_predictor.log')
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Titanic Survival Predictor - ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                    # Train models and select best one
  python main.py predict                  # Generate predictions for test set
  python main.py evaluate                 # Evaluate model performance
  python main.py pipeline                 # Run complete pipeline
  python main.py explore                  # Perform data exploration only
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['train', 'predict', 'evaluate', 'pipeline', 'explore'],
        help='Execution mode for the predictor'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to saved model file (for predict/evaluate modes)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory for output files (default: outputs)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='titanic',
        help='Directory containing train.csv and test.csv (default: titanic)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=MODEL_CONFIG['random_state'],
        help=f'Random seed for reproducibility (default: {MODEL_CONFIG["random_state"]})'
    )
    
    parser.add_argument(
        '--min-accuracy',
        type=float,
        default=MODEL_CONFIG['min_accuracy_threshold'],
        help=f'Minimum accuracy threshold (default: {MODEL_CONFIG["min_accuracy_threshold"]})'
    )
    
    return parser.parse_args()


def run_exploration():
    """Run data exploration and analysis."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data exploration...")
    
    try:
        from data.loader import DataLoader
        from data.explorer import DataExplorer
        from utils.config import DATA_DIR, VISUALIZATIONS_DIR
        
        # Initialize components
        loader = DataLoader(str(DATA_DIR))
        explorer = DataExplorer()
        
        # Load training data
        train_data = loader.load_train_data()
        logger.info(f"Loaded {len(train_data)} training records")
        
        # Generate comprehensive exploration report
        exploration_report = explorer.create_exploration_report(train_data)
        
        # Log key findings
        logger.info(f"Dataset shape: {exploration_report['summary_statistics']['dataset_info']['total_records']} rows, "
                   f"{exploration_report['summary_statistics']['dataset_info']['total_features']} columns")
        
        if 'survival_patterns' in exploration_report:
            survival_rate = exploration_report['survival_patterns']['overall_survival']['survival_rate']
            logger.info(f"Overall survival rate: {survival_rate:.2f}%")
        
        # Log missing values summary
        missing_summary = exploration_report['missing_values']['summary']
        logger.info(f"Columns with missing values: {missing_summary['columns_with_missing']}")
        
        logger.info("Data exploration completed successfully")
        return exploration_report
        
    except Exception as e:
        logger.error(f"Data exploration failed: {str(e)}")
        raise


def run_training():
    """Run model training and selection."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
    
    try:
        from data.loader import DataLoader
        from data.preprocessor import DataPreprocessor
        from models.trainer import ModelTrainer
        from utils.config import DATA_DIR, MODEL_CONFIG, MODELS_DIR
        
        # Initialize components
        loader = DataLoader(str(DATA_DIR))
        preprocessor = DataPreprocessor()
        trainer = ModelTrainer(random_state=MODEL_CONFIG['random_state'])
        
        # Load and preprocess training data
        train_data = loader.load_train_data()
        logger.info(f"Loaded {len(train_data)} training records")
        
        # Fit preprocessor and transform data
        preprocessor.fit(train_data)
        X, y = preprocessor.preprocess_data(train_data, fit_scaler=True)
        logger.info(f"Preprocessed data shape: {X.shape}")
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = trainer.split_data(
            X, y, test_size=MODEL_CONFIG['test_size']
        )
        
        # Train multiple models
        trained_models = trainer.train_multiple_models(X_train, y_train)
        logger.info(f"Trained {len(trained_models)} models")
        
        # Tune hyperparameters
        tuned_models = trainer.tune_hyperparameters(X_train, y_train)
        logger.info("Hyperparameter tuning completed")
        
        # Select best model
        best_name, best_model, best_accuracy, meets_threshold = trainer.select_best_model(
            X_val, y_val, min_accuracy=MODEL_CONFIG['min_accuracy_threshold']
        )
        
        # Save the best model and preprocessor
        model_path = trainer.save_model(best_model, f"best_{best_name}")
        
        # Save preprocessor for later use
        import joblib
        preprocessor_path = MODELS_DIR / "preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")
        
        logger.info(f"Model training completed. Best model: {best_name} (accuracy: {best_accuracy:.4f})")
        return best_name, best_model, best_accuracy, meets_threshold
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise


def run_prediction(model_path: str = None):
    """Generate predictions for test dataset."""
    logger = logging.getLogger(__name__)
    logger.info("Starting prediction generation...")
    
    try:
        from data.loader import DataLoader
        from data.preprocessor import DataPreprocessor
        from models.predictor import Predictor
        from utils.config import DATA_DIR, MODELS_DIR, PREDICTIONS_DIR
        import joblib
        
        # Initialize components
        loader = DataLoader(str(DATA_DIR))
        predictor = Predictor()
        
        # Load test data
        test_data = loader.load_test_data()
        logger.info(f"Loaded {len(test_data)} test records")
        
        # Load preprocessor
        preprocessor_path = MODELS_DIR / "preprocessor.joblib"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Run training first.")
        
        preprocessor = joblib.load(preprocessor_path)
        logger.info("Preprocessor loaded")
        
        # Load model
        if model_path:
            predictor.load_model(model_path)
        else:
            # Find the best model file
            model_files = list(MODELS_DIR.glob("best_*.joblib"))
            if not model_files:
                raise FileNotFoundError("No trained model found. Run training first.")
            
            # Use the most recent model
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            predictor.load_model(str(latest_model))
            logger.info(f"Using model: {latest_model.name}")
        
        # Preprocess test data
        X_test, _ = preprocessor.preprocess_data(test_data, target_col=None, fit_scaler=False)
        logger.info(f"Preprocessed test data shape: {X_test.shape}")
        
        # Generate predictions
        passenger_ids = test_data['PassengerId']
        predictions = predictor.generate_predictions(X_test, passenger_ids)
        
        # Create submission file
        submission_path = PREDICTIONS_DIR / "submission.csv"
        predictor.create_submission_file(predictions, passenger_ids, str(submission_path))
        
        # Validate submission
        validation_report = predictor.validate_competition_compliance(str(submission_path))
        
        if validation_report['competition_compliant']:
            logger.info("✅ Submission file is competition compliant")
        else:
            logger.warning("⚠️ Submission file has validation issues")
            logger.warning(f"Issues: {validation_report['details']}")
        
        # Log prediction summary
        summary = predictor.get_prediction_summary()
        logger.info(f"Prediction summary: {summary['survivors']} survivors out of {summary['total_predictions']} passengers")
        logger.info(f"Survival rate: {summary['survival_rate']:.2%}")
        
        logger.info("Prediction generation completed successfully")
        return predictions, str(submission_path)
        
    except Exception as e:
        logger.error(f"Prediction generation failed: {str(e)}")
        raise


def run_evaluation(model_path: str = None):
    """Evaluate model performance."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation...")
    
    try:
        from data.loader import DataLoader
        from data.preprocessor import DataPreprocessor
        from models.evaluator import ModelEvaluator
        from utils.config import DATA_DIR, MODELS_DIR, VISUALIZATIONS_DIR
        import joblib
        
        # Initialize components
        loader = DataLoader(str(DATA_DIR))
        evaluator = ModelEvaluator()
        
        # Load training data for evaluation
        train_data = loader.load_train_data()
        logger.info(f"Loaded {len(train_data)} training records for evaluation")
        
        # Load preprocessor
        preprocessor_path = MODELS_DIR / "preprocessor.joblib"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Run training first.")
        
        preprocessor = joblib.load(preprocessor_path)
        logger.info("Preprocessor loaded")
        
        # Load model
        if model_path:
            model = joblib.load(model_path)
            model_name = Path(model_path).stem
        else:
            # Find the best model file
            model_files = list(MODELS_DIR.glob("best_*.joblib"))
            if not model_files:
                raise FileNotFoundError("No trained model found. Run training first.")
            
            # Use the most recent model
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            model = joblib.load(str(latest_model))
            model_name = latest_model.stem
            logger.info(f"Using model: {latest_model.name}")
        
        # Preprocess training data
        X, y = preprocessor.preprocess_data(train_data, fit_scaler=False)
        logger.info(f"Preprocessed data shape: {X.shape}")
        
        # Split data for evaluation (same split as training)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Generate predictions on validation set
        y_pred = model.predict(X_val)
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of survival
        
        # Calculate comprehensive metrics
        metrics = evaluator.calculate_metrics(y_val, y_pred)
        logger.info(f"Model performance metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.capitalize()}: {value:.4f}")
        
        # Generate confusion matrix
        cm = evaluator.generate_confusion_matrix(y_val, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Create confusion matrix visualization
        cm_plot_path = VISUALIZATIONS_DIR / f"{model_name}_confusion_matrix.png"
        evaluator.visualize_confusion_matrix(y_val, y_pred, str(cm_plot_path))
        logger.info(f"Confusion matrix plot saved to {cm_plot_path}")
        
        # Analyze feature importance (if model supports it)
        if hasattr(model, 'feature_importances_'):
            feature_names = preprocessor.get_feature_names()
            feature_importance = evaluator.analyze_feature_importance(model, feature_names)
            
            logger.info("Top 10 most important features:")
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
                logger.info(f"  {i}. {feature}: {importance:.4f}")
        
        # Generate classification report
        classification_report = evaluator.generate_classification_report(y_val, y_pred)
        logger.info(f"Classification Report:\n{classification_report}")
        
        # Calculate ROC metrics if probabilities available
        if y_pred_proba is not None:
            roc_metrics = evaluator.calculate_roc_metrics(y_val, y_pred_proba)
            logger.info(f"ROC AUC Score: {roc_metrics['auc']:.4f}")
        
        # Check if model meets accuracy threshold
        accuracy_threshold = 0.8  # From requirements
        meets_threshold = metrics['accuracy'] >= accuracy_threshold
        
        if meets_threshold:
            logger.info(f"✅ Model meets accuracy threshold ({accuracy_threshold:.1%})")
        else:
            logger.warning(f"⚠️ Model does not meet accuracy threshold ({accuracy_threshold:.1%})")
            logger.warning("Consider feature engineering improvements or trying different algorithms")
        
        logger.info("Model evaluation completed successfully")
        return metrics, meets_threshold
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise


def run_pipeline():
    """Run the complete ML pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Starting complete ML pipeline...")
    
    try:
        # Phase 1: Data Exploration
        logger.info("=" * 50)
        logger.info("PHASE 1: DATA EXPLORATION")
        logger.info("=" * 50)
        exploration_report = run_exploration()
        
        # Phase 2: Model Training
        logger.info("=" * 50)
        logger.info("PHASE 2: MODEL TRAINING")
        logger.info("=" * 50)
        best_name, best_model, best_accuracy, meets_threshold = run_training()
        
        # Phase 3: Prediction Generation
        logger.info("=" * 50)
        logger.info("PHASE 3: PREDICTION GENERATION")
        logger.info("=" * 50)
        predictions, submission_path = run_prediction()
        
        # Phase 4: Model Evaluation
        logger.info("=" * 50)
        logger.info("PHASE 4: MODEL EVALUATION")
        logger.info("=" * 50)
        metrics, evaluation_meets_threshold = run_evaluation()
        
        # Pipeline Summary
        logger.info("=" * 50)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"✅ Data exploration completed - {exploration_report['summary_statistics']['dataset_info']['total_records']} records analyzed")
        logger.info(f"✅ Model training completed - Best model: {best_name} (accuracy: {best_accuracy:.4f})")
        logger.info(f"✅ Predictions generated - {len(predictions)} predictions created")
        logger.info(f"✅ Model evaluation completed - Final accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"📁 Submission file: {submission_path}")
        
        if meets_threshold and evaluation_meets_threshold:
            logger.info("🎉 Pipeline completed successfully - Model meets all requirements!")
        else:
            logger.warning("⚠️ Pipeline completed with warnings - Model may need improvement")
        
        logger.info("Complete ML pipeline finished successfully.")
        return {
            'exploration_report': exploration_report,
            'best_model_name': best_name,
            'best_accuracy': best_accuracy,
            'predictions': predictions,
            'submission_path': submission_path,
            'final_metrics': metrics,
            'meets_threshold': meets_threshold and evaluation_meets_threshold
        }
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


def main():
    """Main entry point for the Titanic Survival Predictor."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Ensure output directories exist
    ensure_directories()
    
    logger.info(f"Starting Titanic Survival Predictor in {args.mode} mode")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Random seed: {args.random_seed}")
    
    try:
        # Route to appropriate execution mode
        if args.mode == 'explore':
            run_exploration()
        elif args.mode == 'train':
            run_training()
        elif args.mode == 'predict':
            run_prediction(args.model_path)
        elif args.mode == 'evaluate':
            run_evaluation(args.model_path)
        elif args.mode == 'pipeline':
            run_pipeline()
        
        logger.info("Execution completed successfully.")
        
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()