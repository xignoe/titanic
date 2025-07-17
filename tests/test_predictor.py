"""
Unit tests for the Predictor class.

Tests prediction generation, submission file creation, and validation functionality.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestClassifier

from src.models.predictor import Predictor


class TestPredictor:
    """Test cases for the Predictor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = Predictor()
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.predict.return_value = np.array([0, 1, 0, 1, 1])
        
        # Create test data
        self.test_features = pd.DataFrame({
            'Pclass': [3, 1, 3, 1, 2],
            'Age': [22.0, 38.0, 26.0, 35.0, 35.0],
            'Sex_male': [1, 0, 0, 0, 1],
            'Sex_female': [0, 1, 1, 1, 0],
            'Fare': [7.25, 71.28, 7.92, 53.10, 8.05]
        })
        
        self.passenger_ids = pd.Series([892, 893, 894, 895, 896])
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_init_without_model(self):
        """Test Predictor initialization without model."""
        predictor = Predictor()
        assert predictor.model is None
        assert predictor.predictions is None
        assert predictor.passenger_ids is None
    
    def test_init_with_model(self):
        """Test Predictor initialization with model."""
        predictor = Predictor(self.mock_model)
        assert predictor.model == self.mock_model
    
    def test_set_model(self):
        """Test setting model after initialization."""
        self.predictor.set_model(self.mock_model)
        assert self.predictor.model == self.mock_model
    
    def test_generate_predictions_success(self):
        """Test successful prediction generation."""
        self.predictor.set_model(self.mock_model)
        
        predictions = self.predictor.generate_predictions(self.test_features, self.passenger_ids)
        
        # Check predictions are returned correctly
        expected_predictions = np.array([0, 1, 0, 1, 1])
        np.testing.assert_array_equal(predictions, expected_predictions)
        
        # Check predictions are stored
        np.testing.assert_array_equal(self.predictor.predictions, expected_predictions)
        pd.testing.assert_series_equal(self.predictor.passenger_ids, self.passenger_ids)
        
        # Check model was called correctly
        self.mock_model.predict.assert_called_once()
    
    def test_generate_predictions_no_model(self):
        """Test prediction generation without model raises error."""
        with pytest.raises(ValueError, match="Model must be set"):
            self.predictor.generate_predictions(self.test_features)
    
    def test_generate_predictions_invalid_binary(self):
        """Test prediction generation with non-binary predictions."""
        # Mock model that returns invalid predictions
        invalid_model = Mock()
        invalid_model.predict.return_value = np.array([0, 1, 2, 1, 0])  # Contains 2
        
        self.predictor.set_model(invalid_model)
        
        with pytest.raises(ValueError, match="Predictions must be binary"):
            self.predictor.generate_predictions(self.test_features)
    
    def test_generate_predictions_with_nan(self):
        """Test prediction generation with NaN values."""
        # Mock model that returns NaN
        nan_model = Mock()
        nan_model.predict.return_value = np.array([0, 1, np.nan, 1, 0])
        
        self.predictor.set_model(nan_model)
        
        with pytest.raises(ValueError, match="Predictions contain NaN values"):
            self.predictor.generate_predictions(self.test_features)
    
    def test_validate_predictions_valid_binary(self):
        """Test validation of valid binary predictions."""
        predictions = np.array([0, 1, 0, 1, 1])
        # Should not raise any exception
        self.predictor._validate_predictions(predictions)
    
    def test_validate_predictions_all_zeros(self):
        """Test validation with all zero predictions."""
        predictions = np.array([0, 0, 0, 0, 0])
        # Should not raise any exception
        self.predictor._validate_predictions(predictions)
    
    def test_validate_predictions_all_ones(self):
        """Test validation with all one predictions."""
        predictions = np.array([1, 1, 1, 1, 1])
        # Should not raise any exception
        self.predictor._validate_predictions(predictions)
    
    def test_create_submission_file_success(self):
        """Test successful submission file creation."""
        predictions = np.array([0, 1, 0, 1, 1])
        output_path = os.path.join(self.temp_dir, "test_submission.csv")
        
        result_path = self.predictor.create_submission_file(
            predictions, self.passenger_ids, output_path
        )
        
        # Check file was created
        assert os.path.exists(output_path)
        assert result_path == output_path
        
        # Check file contents
        df = pd.read_csv(output_path)
        assert list(df.columns) == ['PassengerId', 'Survived']
        assert len(df) == 5
        pd.testing.assert_series_equal(df['PassengerId'], self.passenger_ids, check_names=False)
        np.testing.assert_array_equal(df['Survived'].values, predictions)
    
    def test_create_submission_file_with_stored_data(self):
        """Test submission file creation using stored predictions."""
        # Store predictions and passenger IDs
        self.predictor.predictions = np.array([0, 1, 0, 1, 1])
        self.predictor.passenger_ids = self.passenger_ids
        
        output_path = os.path.join(self.temp_dir, "stored_submission.csv")
        
        result_path = self.predictor.create_submission_file(output_path=output_path)
        
        # Check file was created correctly
        assert os.path.exists(output_path)
        df = pd.read_csv(output_path)
        assert len(df) == 5
        assert list(df.columns) == ['PassengerId', 'Survived']
    
    def test_create_submission_file_no_predictions(self):
        """Test submission file creation without predictions."""
        output_path = os.path.join(self.temp_dir, "no_predictions.csv")
        
        with pytest.raises(ValueError, match="No predictions available"):
            self.predictor.create_submission_file(output_path=output_path)
    
    def test_create_submission_file_no_passenger_ids(self):
        """Test submission file creation without passenger IDs."""
        predictions = np.array([0, 1, 0, 1, 1])
        output_path = os.path.join(self.temp_dir, "no_ids.csv")
        
        with pytest.raises(ValueError, match="No passenger IDs available"):
            self.predictor.create_submission_file(predictions, output_path=output_path)
    
    def test_create_submission_file_mismatched_lengths(self):
        """Test submission file creation with mismatched prediction and ID lengths."""
        predictions = np.array([0, 1, 0])  # 3 predictions
        passenger_ids = pd.Series([892, 893, 894, 895])  # 4 IDs
        output_path = os.path.join(self.temp_dir, "mismatched.csv")
        
        with pytest.raises(ValueError, match="must have same length"):
            self.predictor.create_submission_file(predictions, passenger_ids, output_path)
    
    def test_validate_submission_format_valid_file(self):
        """Test validation of valid submission file."""
        # Create valid submission file
        submission_df = pd.DataFrame({
            'PassengerId': range(892, 892 + 418),
            'Survived': np.random.choice([0, 1], 418)
        })
        
        output_path = os.path.join(self.temp_dir, "valid_submission.csv")
        submission_df.to_csv(output_path, index=False)
        
        # Should not raise any exception
        result = self.predictor.validate_submission_format(output_path)
        assert result is True
    
    def test_validate_submission_format_file_not_found(self):
        """Test validation with non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.predictor.validate_submission_format("nonexistent.csv")
    
    def test_validate_submission_format_wrong_columns(self):
        """Test validation with wrong column names."""
        # Create file with wrong columns
        df = pd.DataFrame({
            'ID': range(892, 892 + 418),
            'Prediction': np.random.choice([0, 1], 418)
        })
        
        output_path = os.path.join(self.temp_dir, "wrong_columns.csv")
        df.to_csv(output_path, index=False)
        
        with pytest.raises(ValueError, match="Columns must be"):
            self.predictor.validate_submission_format(output_path)
    
    def test_validate_submission_format_wrong_row_count(self):
        """Test validation with wrong number of rows."""
        # Create file with wrong number of rows
        df = pd.DataFrame({
            'PassengerId': range(892, 892 + 100),  # Only 100 rows instead of 418
            'Survived': np.random.choice([0, 1], 100)
        })
        
        output_path = os.path.join(self.temp_dir, "wrong_rows.csv")
        df.to_csv(output_path, index=False)
        
        with pytest.raises(ValueError, match="exactly 418 data rows"):
            self.predictor.validate_submission_format(output_path)
    
    def test_validate_submission_format_non_binary_survived(self):
        """Test validation with non-binary Survived values."""
        # Create file with invalid Survived values
        df = pd.DataFrame({
            'PassengerId': range(892, 892 + 418),
            'Survived': np.random.choice([0, 1, 2], 418)  # Contains 2
        })
        
        output_path = os.path.join(self.temp_dir, "non_binary.csv")
        df.to_csv(output_path, index=False)
        
        with pytest.raises(ValueError, match="only 0 or 1 values"):
            self.predictor.validate_submission_format(output_path)
    
    def test_validate_submission_format_duplicate_passenger_ids(self):
        """Test validation with duplicate PassengerId values."""
        # Create file with duplicate IDs
        passenger_ids = list(range(892, 892 + 417)) + [892]  # Duplicate 892
        df = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Survived': np.random.choice([0, 1], 418)
        })
        
        output_path = os.path.join(self.temp_dir, "duplicate_ids.csv")
        df.to_csv(output_path, index=False)
        
        with pytest.raises(ValueError, match="must be unique"):
            self.predictor.validate_submission_format(output_path)
    
    def test_validate_submission_format_missing_values(self):
        """Test validation with missing values."""
        # Create file with missing values
        df = pd.DataFrame({
            'PassengerId': range(892, 892 + 418),
            'Survived': [np.nan] + list(np.random.choice([0, 1], 417))
        })
        
        output_path = os.path.join(self.temp_dir, "missing_values.csv")
        df.to_csv(output_path, index=False)
        
        with pytest.raises(ValueError, match="cannot contain missing values"):
            self.predictor.validate_submission_format(output_path)
    
    def test_predict_and_submit_success(self):
        """Test combined prediction and submission generation."""
        self.predictor.set_model(self.mock_model)
        output_path = os.path.join(self.temp_dir, "combined_submission.csv")
        
        predictions, submission_path = self.predictor.predict_and_submit(
            self.test_features, self.passenger_ids, output_path
        )
        
        # Check predictions
        expected_predictions = np.array([0, 1, 0, 1, 1])
        np.testing.assert_array_equal(predictions, expected_predictions)
        
        # Check submission file
        assert submission_path == output_path
        assert os.path.exists(output_path)
        
        df = pd.read_csv(output_path)
        assert len(df) == 5
        assert list(df.columns) == ['PassengerId', 'Survived']
    
    def test_get_prediction_summary_success(self):
        """Test getting prediction summary."""
        predictions = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        self.predictor.predictions = predictions
        
        summary = self.predictor.get_prediction_summary()
        
        expected_summary = {
            'total_predictions': 8,
            'survivors': 4,
            'non_survivors': 4,
            'survival_rate': 0.5,
            'prediction_distribution': {
                'died': 4,
                'survived': 4
            }
        }
        
        assert summary == expected_summary
    
    def test_get_prediction_summary_no_predictions(self):
        """Test getting summary without predictions."""
        with pytest.raises(ValueError, match="No predictions available"):
            self.predictor.get_prediction_summary()
    
    def test_load_model_success(self):
        """Test successful model loading."""
        # Create a simple model and save it
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.test_features, [0, 1, 0, 1, 1])
        
        model_path = os.path.join(self.temp_dir, "test_model.joblib")
        import joblib
        joblib.dump(model, model_path)
        
        # Load the model
        loaded_model = self.predictor.load_model(model_path)
        
        assert self.predictor.model is not None
        assert loaded_model is not None
        
        # Test that loaded model can make predictions
        predictions = loaded_model.predict(self.test_features)
        assert len(predictions) == len(self.test_features)
    
    def test_load_model_file_not_found(self):
        """Test model loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.predictor.load_model("nonexistent_model.joblib")
    
    def test_prediction_count_validation_warning(self):
        """Test that warning is logged for non-418 predictions."""
        # Create model that returns different number of predictions
        short_model = Mock()
        short_model.predict.return_value = np.array([0, 1, 0])  # Only 3 predictions
        
        self.predictor.set_model(short_model)
        
        # Should still work but log warning
        with patch('src.models.predictor.logger') as mock_logger:
            predictions = self.predictor.generate_predictions(self.test_features.iloc[:3])
            mock_logger.warning.assert_called_with("Expected 418 predictions, got 3")
    
    def test_real_titanic_size_predictions(self):
        """Test with realistic Titanic test set size (418 samples)."""
        # Create 418 test samples
        large_features = pd.DataFrame({
            'Pclass': np.random.choice([1, 2, 3], 418),
            'Age': np.random.uniform(0, 80, 418),
            'Sex_male': np.random.choice([0, 1], 418),
            'Sex_female': np.random.choice([0, 1], 418),
            'Fare': np.random.uniform(0, 100, 418)
        })
        
        large_passenger_ids = pd.Series(range(892, 892 + 418))
        
        # Mock model for 418 predictions
        large_model = Mock()
        large_model.predict.return_value = np.random.choice([0, 1], 418)
        
        self.predictor.set_model(large_model)
        
        predictions = self.predictor.generate_predictions(large_features, large_passenger_ids)
        
        assert len(predictions) == 418
        assert all(pred in [0, 1] for pred in predictions)
        
        # Create submission file
        output_path = os.path.join(self.temp_dir, "titanic_submission.csv")
        submission_path = self.predictor.create_submission_file(
            predictions, large_passenger_ids, output_path
        )
        
        # Validate submission format
        assert self.predictor.validate_submission_format(submission_path) is True
    
    def test_validate_competition_compliance_fully_compliant(self):
        """Test comprehensive competition compliance validation with fully compliant file."""
        # Create fully compliant submission file
        submission_df = pd.DataFrame({
            'PassengerId': range(892, 892 + 418),
            'Survived': np.random.choice([0, 1], 418)
        })
        
        output_path = os.path.join(self.temp_dir, "compliant_submission.csv")
        submission_df.to_csv(output_path, index=False)
        
        # Test compliance validation
        report = self.predictor.validate_competition_compliance(output_path)
        
        # Check all validation flags are True
        assert report['file_exists'] is True
        assert report['correct_format'] is True
        assert report['correct_columns'] is True
        assert report['correct_row_count'] is True
        assert report['binary_predictions'] is True
        assert report['no_missing_values'] is True
        assert report['unique_passenger_ids'] is True
        assert report['csv_format'] is True
        assert report['competition_compliant'] is True
        
        # Check prediction summary is included
        assert 'prediction_summary' in report['details']
        assert report['details']['prediction_summary']['total'] == 418
    
    def test_validate_competition_compliance_non_csv_file(self):
        """Test compliance validation with non-CSV file."""
        # Create non-CSV file
        output_path = os.path.join(self.temp_dir, "submission.txt")
        with open(output_path, 'w') as f:
            f.write("PassengerId,Survived\n892,1\n")
        
        report = self.predictor.validate_competition_compliance(output_path)
        
        assert report['csv_format'] is False
        assert report['competition_compliant'] is False
        assert 'format_error' in report['details']
    
    def test_validate_competition_compliance_wrong_columns(self):
        """Test compliance validation with wrong column names."""
        # Create file with wrong columns
        df = pd.DataFrame({
            'ID': range(892, 892 + 418),
            'Prediction': np.random.choice([0, 1], 418)
        })
        
        output_path = os.path.join(self.temp_dir, "wrong_columns_compliance.csv")
        df.to_csv(output_path, index=False)
        
        report = self.predictor.validate_competition_compliance(output_path)
        
        assert report['correct_columns'] is False
        assert report['competition_compliant'] is False
        assert 'column_error' in report['details']
    
    def test_validate_competition_compliance_wrong_row_count(self):
        """Test compliance validation with wrong number of rows."""
        # Create file with wrong number of rows
        df = pd.DataFrame({
            'PassengerId': range(892, 892 + 100),
            'Survived': np.random.choice([0, 1], 100)
        })
        
        output_path = os.path.join(self.temp_dir, "wrong_rows_compliance.csv")
        df.to_csv(output_path, index=False)
        
        report = self.predictor.validate_competition_compliance(output_path)
        
        assert report['correct_row_count'] is False
        assert report['competition_compliant'] is False
        assert 'row_count_error' in report['details']
    
    def test_validate_competition_compliance_invalid_predictions(self):
        """Test compliance validation with invalid prediction values."""
        # Create file with invalid predictions
        df = pd.DataFrame({
            'PassengerId': range(892, 892 + 418),
            'Survived': np.random.choice([0, 1, 2], 418)  # Contains invalid value 2
        })
        
        output_path = os.path.join(self.temp_dir, "invalid_predictions_compliance.csv")
        df.to_csv(output_path, index=False)
        
        report = self.predictor.validate_competition_compliance(output_path)
        
        assert report['binary_predictions'] is False
        assert report['competition_compliant'] is False
        assert 'invalid_predictions' in report['details']
    
    def test_validate_competition_compliance_missing_values(self):
        """Test compliance validation with missing values."""
        # Create file with missing values
        df = pd.DataFrame({
            'PassengerId': range(892, 892 + 418),
            'Survived': [np.nan] + list(np.random.choice([0, 1], 417))
        })
        
        output_path = os.path.join(self.temp_dir, "missing_values_compliance.csv")
        df.to_csv(output_path, index=False)
        
        report = self.predictor.validate_competition_compliance(output_path)
        
        assert report['no_missing_values'] is False
        assert report['competition_compliant'] is False
        assert 'missing_values' in report['details']
    
    def test_validate_competition_compliance_duplicate_ids(self):
        """Test compliance validation with duplicate PassengerId values."""
        # Create file with duplicate IDs
        passenger_ids = list(range(892, 892 + 417)) + [892]  # Duplicate 892
        df = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Survived': np.random.choice([0, 1], 418)
        })
        
        output_path = os.path.join(self.temp_dir, "duplicate_ids_compliance.csv")
        df.to_csv(output_path, index=False)
        
        report = self.predictor.validate_competition_compliance(output_path)
        
        assert report['unique_passenger_ids'] is False
        assert report['competition_compliant'] is False
        assert 'duplicate_ids' in report['details']
        assert 892 in report['details']['duplicate_ids']
    
    def test_validate_competition_compliance_empty_file(self):
        """Test compliance validation with empty file."""
        # Create empty CSV file
        output_path = os.path.join(self.temp_dir, "empty_compliance.csv")
        with open(output_path, 'w') as f:
            f.write("")
        
        report = self.predictor.validate_competition_compliance(output_path)
        
        assert report['competition_compliant'] is False
        assert 'error' in report['details']
    
    def test_validate_competition_compliance_file_not_found(self):
        """Test compliance validation with non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.predictor.validate_competition_compliance("nonexistent_compliance.csv")
    
    def test_submission_file_exactly_418_rows_plus_header(self):
        """Test that submission file has exactly 418 data rows plus 1 header row."""
        # Create 418 predictions
        predictions = np.random.choice([0, 1], 418)
        passenger_ids = pd.Series(range(892, 892 + 418))
        
        output_path = os.path.join(self.temp_dir, "exact_rows_submission.csv")
        
        self.predictor.create_submission_file(predictions, passenger_ids, output_path)
        
        # Read file and count lines manually
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        # Should have exactly 419 lines (1 header + 418 data rows)
        assert len(lines) == 419
        
        # First line should be header
        assert lines[0].strip() == "PassengerId,Survived"
        
        # Validate using pandas
        df = pd.read_csv(output_path)
        assert len(df) == 418  # Data rows only
        assert list(df.columns) == ['PassengerId', 'Survived']
    
    def test_submission_file_csv_format_specification(self):
        """Test that submission file matches exact CSV format specification."""
        predictions = np.array([0, 1, 0, 1, 1])
        passenger_ids = pd.Series([892, 893, 894, 895, 896])
        
        output_path = os.path.join(self.temp_dir, "format_spec_submission.csv")
        
        self.predictor.create_submission_file(predictions, passenger_ids, output_path)
        
        # Read file content as text to check exact format
        with open(output_path, 'r') as f:
            content = f.read()
        
        expected_lines = [
            "PassengerId,Survived",
            "892,0",
            "893,1", 
            "894,0",
            "895,1",
            "896,1"
        ]
        
        actual_lines = content.strip().split('\n')
        assert actual_lines == expected_lines
        
        # Ensure no extra whitespace or formatting issues
        for line in actual_lines[1:]:  # Skip header
            parts = line.split(',')
            assert len(parts) == 2
            assert parts[0].isdigit()  # PassengerId is numeric
            assert parts[1] in ['0', '1']  # Survived is binary
    
    def test_validate_submission_format_strict_vs_non_strict(self):
        """Test validation with strict and non-strict row count checking."""
        # Create file with non-418 rows
        df = pd.DataFrame({
            'PassengerId': range(892, 892 + 100),
            'Survived': np.random.choice([0, 1], 100)
        })
        
        output_path = os.path.join(self.temp_dir, "non_standard_rows.csv")
        df.to_csv(output_path, index=False)
        
        # Strict validation should raise error
        with pytest.raises(ValueError, match="exactly 418 data rows"):
            self.predictor.validate_submission_format(output_path, strict_row_count=True)
        
        # Non-strict validation should pass but log warning
        with patch('src.models.predictor.logger') as mock_logger:
            result = self.predictor.validate_submission_format(output_path, strict_row_count=False)
            assert result is True
            mock_logger.warning.assert_called_with("Non-standard row count: 100 (expected 418 for Kaggle)")
    
    def test_create_submission_file_directory_creation(self):
        """Test that submission file creation creates necessary directories."""
        predictions = np.array([0, 1, 0])
        passenger_ids = pd.Series([892, 893, 894])
        
        # Use nested directory path that doesn't exist
        nested_path = os.path.join(self.temp_dir, "deep", "nested", "path", "submission.csv")
        
        result_path = self.predictor.create_submission_file(predictions, passenger_ids, nested_path)
        
        # Check file was created and directories exist
        assert os.path.exists(result_path)
        assert os.path.exists(os.path.dirname(nested_path))
        
        # Validate file content
        df = pd.read_csv(result_path)
        assert len(df) == 3
        assert list(df.columns) == ['PassengerId', 'Survived']