"""
Unit tests for the ModelTrainer class.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sys
import os
import tempfile
import shutil
import joblib

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.trainer import ModelTrainer


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample classification data for testing."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Convert to DataFrame and Series
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series
    
    @pytest.fixture
    def trainer(self):
        """Create ModelTrainer instance for testing."""
        return ModelTrainer(random_state=42)
    
    def test_init(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(random_state=123)
        assert trainer.random_state == 123
        assert trainer.models == {}
        assert trainer.model_scores == {}
        assert trainer.tuned_models == {}
        assert trainer.best_params == {}
    
    def test_split_data(self, trainer, sample_data):
        """Test data splitting functionality."""
        X, y = sample_data
        
        X_train, X_val, y_train, y_val = trainer.split_data(X, y, test_size=0.2)
        
        # Check sizes
        assert len(X_train) == 160  # 80% of 200
        assert len(X_val) == 40     # 20% of 200
        assert len(y_train) == 160
        assert len(y_val) == 40
        
        # Check that indices don't overlap
        train_indices = set(X_train.index)
        val_indices = set(X_val.index)
        assert len(train_indices.intersection(val_indices)) == 0
        
        # Check stratification (approximately equal class distribution)
        train_class_ratio = y_train.mean()
        val_class_ratio = y_val.mean()
        assert abs(train_class_ratio - val_class_ratio) < 0.1
    
    def test_split_data_no_stratify(self, trainer, sample_data):
        """Test data splitting without stratification."""
        X, y = sample_data
        
        X_train, X_val, y_train, y_val = trainer.split_data(X, y, test_size=0.3, stratify=False)
        
        # Check sizes
        assert len(X_train) == 140  # 70% of 200
        assert len(X_val) == 60     # 30% of 200
    
    def test_get_default_models(self, trainer):
        """Test default model configuration."""
        models = trainer._get_default_models()
        
        assert len(models) == 3
        assert 'RandomForest' in models
        assert 'LogisticRegression' in models
        assert 'SVM' in models
        
        assert isinstance(models['RandomForest'], RandomForestClassifier)
        assert isinstance(models['LogisticRegression'], LogisticRegression)
        assert isinstance(models['SVM'], SVC)
    
    def test_train_multiple_models(self, trainer, sample_data):
        """Test training multiple models."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = trainer.split_data(X, y)
        
        trained_models = trainer.train_multiple_models(X_train, y_train)
        
        # Check that all models were trained
        assert len(trained_models) == 3
        assert 'RandomForest' in trained_models
        assert 'LogisticRegression' in trained_models
        assert 'SVM' in trained_models
        
        # Check that models are fitted
        for name, model in trained_models.items():
            assert hasattr(model, 'predict')
            # Test that model can make predictions
            predictions = model.predict(X_val)
            assert len(predictions) == len(X_val)
            assert all(pred in [0, 1] for pred in predictions)
    
    def test_train_custom_models(self, trainer, sample_data):
        """Test training with custom model configuration."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = trainer.split_data(X, y)
        
        custom_models = {
            'CustomRF': RandomForestClassifier(n_estimators=50, random_state=42),
            'CustomLR': LogisticRegression(C=0.5, random_state=42)
        }
        
        trained_models = trainer.train_multiple_models(X_train, y_train, custom_models)
        
        assert len(trained_models) == 2
        assert 'CustomRF' in trained_models
        assert 'CustomLR' in trained_models
    
    def test_perform_cross_validation(self, trainer, sample_data):
        """Test cross-validation functionality."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = trainer.split_data(X, y)
        
        # Train a single model for testing
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        cv_results = trainer.perform_cross_validation(model, X_train, y_train, cv_folds=3)
        
        # Check results structure
        assert 'cv_scores' in cv_results
        assert 'mean_score' in cv_results
        assert 'std_score' in cv_results
        assert 'min_score' in cv_results
        assert 'max_score' in cv_results
        
        # Check that we have the right number of CV scores
        assert len(cv_results['cv_scores']) == 3
        
        # Check that scores are reasonable
        assert 0 <= cv_results['mean_score'] <= 1
        assert cv_results['std_score'] >= 0
        assert cv_results['min_score'] <= cv_results['mean_score'] <= cv_results['max_score']
    
    def test_evaluate_models(self, trainer, sample_data):
        """Test model evaluation functionality."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = trainer.split_data(X, y)
        
        # Train models first
        trainer.train_multiple_models(X_train, y_train)
        
        # Evaluate models
        evaluation_results = trainer.evaluate_models(X_val, y_val)
        
        # Check results structure
        assert len(evaluation_results) == 3
        for model_name in ['RandomForest', 'LogisticRegression', 'SVM']:
            assert model_name in evaluation_results
            
            model_results = evaluation_results[model_name]
            assert 'validation_accuracy' in model_results
            assert 'cv_mean_score' in model_results
            assert 'cv_std_score' in model_results
            
            # Check that accuracy is reasonable
            assert 0 <= model_results['validation_accuracy'] <= 1
            assert 0 <= model_results['cv_mean_score'] <= 1
            assert model_results['cv_std_score'] >= 0
    
    def test_evaluate_models_no_training(self, trainer, sample_data):
        """Test that evaluate_models raises error when no models are trained."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = trainer.split_data(X, y)
        
        with pytest.raises(ValueError, match="No models have been trained yet"):
            trainer.evaluate_models(X_val, y_val)
    
    def test_get_best_model(self, trainer, sample_data):
        """Test getting the best performing model."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = trainer.split_data(X, y)
        
        # Train and evaluate models
        trainer.train_multiple_models(X_train, y_train)
        trainer.evaluate_models(X_val, y_val)
        
        best_name, best_model, best_score = trainer.get_best_model()
        
        # Check that we get valid results
        assert best_name in ['RandomForest', 'LogisticRegression', 'SVM']
        assert hasattr(best_model, 'predict')
        assert 0 <= best_score <= 1
        
        # Check that this is indeed the best score
        all_scores = [scores['validation_accuracy'] for scores in trainer.model_scores.values()]
        assert best_score == max(all_scores)
    
    def test_get_best_model_no_evaluation(self, trainer):
        """Test that get_best_model raises error when no evaluations are available."""
        with pytest.raises(ValueError, match="No model evaluations available"):
            trainer.get_best_model()
    
    def test_get_model_summary(self, trainer, sample_data):
        """Test getting model performance summary."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = trainer.split_data(X, y)
        
        # Train and evaluate models
        trainer.train_multiple_models(X_train, y_train)
        trainer.evaluate_models(X_val, y_val)
        
        summary = trainer.get_model_summary()
        
        # Check DataFrame structure
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 3
        assert 'Model' in summary.columns
        assert 'Validation_Accuracy' in summary.columns
        assert 'CV_Mean_Score' in summary.columns
        assert 'CV_Std_Score' in summary.columns
        
        # Check that it's sorted by validation accuracy (descending)
        accuracies = summary['Validation_Accuracy'].values
        assert all(accuracies[i] >= accuracies[i+1] for i in range(len(accuracies)-1))
    
    def test_get_model_summary_no_evaluation(self, trainer):
        """Test that get_model_summary raises error when no evaluations are available."""
        with pytest.raises(ValueError, match="No model evaluations available"):
            trainer.get_model_summary()
    
    def test_reproducibility(self, sample_data):
        """Test that results are reproducible with same random state."""
        X, y = sample_data
        
        # Train two trainers with same random state
        trainer1 = ModelTrainer(random_state=42)
        trainer2 = ModelTrainer(random_state=42)
        
        # Split data and train models
        X_train1, X_val1, y_train1, y_val1 = trainer1.split_data(X, y)
        X_train2, X_val2, y_train2, y_val2 = trainer2.split_data(X, y)
        
        # Check that splits are identical
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_val1, X_val2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_val1, y_val2)
        
        # Train models and check predictions are identical
        trainer1.train_multiple_models(X_train1, y_train1)
        trainer2.train_multiple_models(X_train2, y_train2)
        
        for model_name in ['RandomForest', 'LogisticRegression', 'SVM']:
            pred1 = trainer1.models[model_name].predict(X_val1)
            pred2 = trainer2.models[model_name].predict(X_val2)
            np.testing.assert_array_equal(pred1, pred2)
    
    def test_get_hyperparameter_grids(self, trainer):
        """Test hyperparameter grid configuration."""
        grids = trainer._get_hyperparameter_grids()
        
        assert len(grids) == 3
        assert 'RandomForest' in grids
        assert 'LogisticRegression' in grids
        assert 'SVM' in grids
        
        # Check that each grid has expected parameters
        assert 'n_estimators' in grids['RandomForest']
        assert 'C' in grids['LogisticRegression']
        assert 'kernel' in grids['SVM']
    
    def test_tune_hyperparameters(self, trainer, sample_data):
        """Test hyperparameter tuning functionality."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = trainer.split_data(X, y)
        
        # Train models first
        trainer.train_multiple_models(X_train, y_train)
        
        # Use smaller parameter grids for faster testing
        small_grids = {
            'RandomForest': {
                'n_estimators': [10, 20],
                'max_depth': [3, 5]
            },
            'LogisticRegression': {
                'C': [0.1, 1.0]
            }
        }
        
        tuned_models = trainer.tune_hyperparameters(X_train, y_train, small_grids, cv_folds=2)
        
        # Check that tuned models were created
        assert len(tuned_models) >= 2
        assert 'RandomForest' in tuned_models
        assert 'LogisticRegression' in tuned_models
        
        # Check that best parameters were stored
        assert len(trainer.best_params) >= 2
        assert 'RandomForest' in trainer.best_params
        assert 'LogisticRegression' in trainer.best_params
        
        # Check that tuned models can make predictions
        for name, model in tuned_models.items():
            predictions = model.predict(X_val)
            assert len(predictions) == len(X_val)
    
    def test_tune_hyperparameters_no_models(self, trainer, sample_data):
        """Test that tune_hyperparameters raises error when no models are trained."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = trainer.split_data(X, y)
        
        with pytest.raises(ValueError, match="No models have been trained yet"):
            trainer.tune_hyperparameters(X_train, y_train)
    
    def test_compare_models(self, trainer, sample_data):
        """Test model comparison functionality."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = trainer.split_data(X, y)
        
        # Train models
        trainer.train_multiple_models(X_train, y_train)
        
        # Compare base models
        comparison_df = trainer.compare_models(X_val, y_val, use_tuned=False)
        
        # Check DataFrame structure
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 3
        assert 'Model' in comparison_df.columns
        assert 'Validation_Accuracy' in comparison_df.columns
        assert 'CV_Mean_Score' in comparison_df.columns
        assert 'Tuned' in comparison_df.columns
        
        # Check that it's sorted by validation accuracy (descending)
        accuracies = comparison_df['Validation_Accuracy'].values
        assert all(accuracies[i] >= accuracies[i+1] for i in range(len(accuracies)-1))
        
        # Check that all models are marked as not tuned
        assert all(not tuned for tuned in comparison_df['Tuned'])
    
    def test_compare_tuned_models(self, trainer, sample_data):
        """Test comparison of tuned models."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = trainer.split_data(X, y)
        
        # Train and tune models
        trainer.train_multiple_models(X_train, y_train)
        
        # Use small grids for faster testing
        small_grids = {
            'RandomForest': {'n_estimators': [10, 20]},
            'LogisticRegression': {'C': [0.1, 1.0]}
        }
        trainer.tune_hyperparameters(X_train, y_train, small_grids, cv_folds=2)
        
        # Compare tuned models
        comparison_df = trainer.compare_models(X_val, y_val, use_tuned=True)
        
        # Check that tuned models are marked correctly
        tuned_models = comparison_df[comparison_df['Tuned'] == True]
        assert len(tuned_models) >= 2
    
    def test_select_best_model(self, trainer, sample_data):
        """Test best model selection."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = trainer.split_data(X, y)
        
        # Train models
        trainer.train_multiple_models(X_train, y_train)
        
        # Select best model
        best_name, best_model, best_accuracy, meets_threshold = trainer.select_best_model(
            X_val, y_val, min_accuracy=0.5, use_tuned=False
        )
        
        # Check results
        assert best_name in ['RandomForest', 'LogisticRegression', 'SVM']
        assert hasattr(best_model, 'predict')
        assert 0 <= best_accuracy <= 1
        assert meets_threshold in [True, False]  # More flexible boolean check
        
        # Test with high threshold
        _, _, _, meets_high_threshold = trainer.select_best_model(
            X_val, y_val, min_accuracy=0.99, use_tuned=False
        )
        assert not meets_high_threshold
    
    def test_save_and_load_model(self, trainer, sample_data):
        """Test model saving and loading functionality."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = trainer.split_data(X, y)
        
        # Train a model
        trainer.train_multiple_models(X_train, y_train)
        model = trainer.models['RandomForest']
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'test_model.joblib')
            
            # Save model
            saved_path = trainer.save_model(model, 'test_model', filepath)
            assert saved_path == filepath
            assert os.path.exists(filepath)
            
            # Load model
            loaded_model = trainer.load_model(filepath)
            
            # Test that loaded model works
            original_pred = model.predict(X_val)
            loaded_pred = loaded_model.predict(X_val)
            np.testing.assert_array_equal(original_pred, loaded_pred)
    
    def test_save_model_default_path(self, trainer, sample_data):
        """Test model saving with default path."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = trainer.split_data(X, y)
        
        # Train a model
        trainer.train_multiple_models(X_train, y_train)
        model = trainer.models['RandomForest']
        
        try:
            # Save model with default path
            saved_path = trainer.save_model(model, 'test_model')
            
            # Check that file was created
            assert os.path.exists(saved_path)
            assert 'outputs/models' in saved_path
            assert 'test_model_model.joblib' in saved_path
            
            # Clean up
            if os.path.exists(saved_path):
                os.remove(saved_path)
            
            # Remove directory if empty
            outputs_dir = 'outputs/models'
            if os.path.exists(outputs_dir) and not os.listdir(outputs_dir):
                os.rmdir(outputs_dir)
                if os.path.exists('outputs') and not os.listdir('outputs'):
                    os.rmdir('outputs')
                    
        except Exception as e:
            # Clean up in case of error
            if 'saved_path' in locals() and os.path.exists(saved_path):
                os.remove(saved_path)
            raise e
    
    def test_save_best_model(self, trainer, sample_data):
        """Test saving the best performing model."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = trainer.split_data(X, y)
        
        # Train models
        trainer.train_multiple_models(X_train, y_train)
        
        try:
            # Save best model
            best_name, saved_path = trainer.save_best_model(X_val, y_val, min_accuracy=0.5)
            
            # Check results
            assert best_name in ['RandomForest', 'LogisticRegression', 'SVM']
            assert os.path.exists(saved_path)
            assert f'best_{best_name}' in saved_path
            
            # Check that metadata file was also created
            metadata_path = saved_path.replace('.joblib', '_metadata.joblib')
            assert os.path.exists(metadata_path)
            
            # Load and check metadata
            metadata = joblib.load(metadata_path)
            assert 'model_name' in metadata
            assert 'accuracy' in metadata
            assert 'meets_threshold' in metadata
            
            # Clean up
            if os.path.exists(saved_path):
                os.remove(saved_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                
            # Remove directory if empty
            outputs_dir = 'outputs/models'
            if os.path.exists(outputs_dir) and not os.listdir(outputs_dir):
                os.rmdir(outputs_dir)
                if os.path.exists('outputs') and not os.listdir('outputs'):
                    os.rmdir('outputs')
                    
        except Exception as e:
            # Clean up in case of error
            for path in ['saved_path', 'metadata_path']:
                if path in locals() and os.path.exists(locals()[path]):
                    os.remove(locals()[path])
            raise e
    
    def test_load_nonexistent_model(self, trainer):
        """Test loading a model that doesn't exist."""
        with pytest.raises(Exception):
            trainer.load_model('nonexistent_model.joblib')
    
    def test_compare_models_no_models(self, trainer, sample_data):
        """Test that compare_models raises error when no models are available."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = trainer.split_data(X, y)
        
        with pytest.raises(ValueError, match="No models available for comparison"):
            trainer.compare_models(X_val, y_val)


if __name__ == "__main__":
    pytest.main([__file__])