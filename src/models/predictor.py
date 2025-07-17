"""
Prediction generation module for the Titanic Survival Predictor.

This module provides the Predictor class that handles generating survival
predictions for the test dataset and creating submission files.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Any, Optional, Tuple
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor:
    """
    Handles prediction generation for test dataset and submission file creation.
    
    This class provides functionality to:
    - Generate survival predictions for test data
    - Validate prediction format and count
    - Create competition-compliant submission files
    - Validate submission file format
    """
    
    def __init__(self, model: Any = None):
        """
        Initialize the Predictor.
        
        Args:
            model: Trained model for making predictions. Can be set later.
        """
        self.model = model
        self.predictions = None
        self.passenger_ids = None
        
    def set_model(self, model: Any) -> None:
        """
        Set the trained model for predictions.
        
        Args:
            model: Trained scikit-learn model
        """
        self.model = model
        logger.info("Model set for prediction")
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            Loaded model instance
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise
    
    def generate_predictions(self, X_test: pd.DataFrame, 
                           passenger_ids: pd.Series = None) -> np.ndarray:
        """
        Generate survival predictions for test dataset.
        
        Args:
            X_test: Preprocessed test features
            passenger_ids: Optional passenger IDs for validation
            
        Returns:
            numpy.ndarray: Binary predictions (0 or 1)
            
        Raises:
            ValueError: If model is not set or predictions are invalid
        """
        if self.model is None:
            raise ValueError("Model must be set before generating predictions. Use set_model() or load_model().")
        
        logger.info(f"Generating predictions for {len(X_test)} test samples")
        
        try:
            # Generate predictions
            predictions = self.model.predict(X_test)
            
            # Validate predictions are binary
            self._validate_predictions(predictions)
            
            # Store predictions and passenger IDs
            self.predictions = predictions
            if passenger_ids is not None:
                self.passenger_ids = passenger_ids
            
            logger.info(f"Successfully generated {len(predictions)} predictions")
            logger.info(f"Prediction distribution: {np.bincount(predictions)}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to generate predictions: {str(e)}")
            raise
    
    def _validate_predictions(self, predictions: np.ndarray) -> None:
        """
        Validate that predictions are in correct format.
        
        Args:
            predictions: Array of predictions to validate
            
        Raises:
            ValueError: If predictions are not valid
        """
        # Check for NaN values first
        if np.isnan(predictions).any():
            raise ValueError("Predictions contain NaN values")
        
        # Check predictions are binary (0 or 1)
        unique_values = np.unique(predictions)
        if not np.array_equal(np.sort(unique_values), np.array([0, 1])) and not (
            len(unique_values) == 1 and unique_values[0] in [0, 1]
        ):
            raise ValueError(f"Predictions must be binary (0 or 1), found: {unique_values}")
        
        # Check for expected count (418 for Titanic competition)
        if len(predictions) != 418:
            logger.warning(f"Expected 418 predictions, got {len(predictions)}")
        
        logger.info("Prediction validation passed")
    
    def create_submission_file(self, predictions: np.ndarray = None, 
                             passenger_ids: pd.Series = None,
                             output_path: str = "outputs/predictions/submission.csv") -> str:
        """
        Create CSV submission file with PassengerId and Survived columns.
        
        Args:
            predictions: Optional predictions array. Uses stored predictions if None.
            passenger_ids: Optional passenger IDs. Uses stored IDs if None.
            output_path: Path where submission file will be saved
            
        Returns:
            str: Path to created submission file
            
        Raises:
            ValueError: If predictions or passenger IDs are missing
        """
        # Use provided or stored predictions
        if predictions is None:
            if self.predictions is None:
                raise ValueError("No predictions available. Generate predictions first.")
            predictions = self.predictions
        
        # Use provided or stored passenger IDs
        if passenger_ids is None:
            if self.passenger_ids is None:
                raise ValueError("No passenger IDs available. Provide passenger_ids parameter.")
            passenger_ids = self.passenger_ids
        
        # Validate inputs
        if len(predictions) != len(passenger_ids):
            raise ValueError(f"Predictions ({len(predictions)}) and passenger IDs ({len(passenger_ids)}) must have same length")
        
        logger.info(f"Creating submission file with {len(predictions)} predictions")
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Survived': predictions.astype(int)
        })
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if path has a directory component
            os.makedirs(output_dir, exist_ok=True)
        
        # Save submission file
        submission_df.to_csv(output_path, index=False)
        
        logger.info(f"Submission file created: {output_path}")
        
        # Validate the created file (with relaxed row count for flexibility)
        self.validate_submission_format(output_path, strict_row_count=False)
        
        return output_path
    
    def validate_submission_format(self, filepath: str, strict_row_count: bool = True) -> bool:
        """
        Validate submission file format against competition requirements.
        
        Args:
            filepath: Path to submission file
            strict_row_count: Whether to enforce exactly 418 rows (True for Kaggle submission)
            
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If validation fails
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Submission file not found: {filepath}")
        
        logger.info(f"Validating submission file format: {filepath}")
        
        try:
            # Load submission file
            df = pd.read_csv(filepath)
            
            # Check number of columns (exactly 2)
            if len(df.columns) != 2:
                raise ValueError(f"Submission must have exactly 2 columns, found {len(df.columns)}")
            
            # Check column names
            expected_columns = ['PassengerId', 'Survived']
            if list(df.columns) != expected_columns:
                raise ValueError(f"Columns must be {expected_columns}, found {list(df.columns)}")
            
            # Check number of rows (exactly 418 for Kaggle submission)
            if strict_row_count and len(df) != 418:
                raise ValueError(f"Submission must have exactly 418 data rows, found {len(df)}")
            elif not strict_row_count and len(df) != 418:
                logger.warning(f"Non-standard row count: {len(df)} (expected 418 for Kaggle)")
            
            # Check PassengerId values
            if not pd.api.types.is_numeric_dtype(df['PassengerId']):
                raise ValueError("PassengerId column must be numeric")
            
            if df['PassengerId'].duplicated().any():
                raise ValueError("PassengerId values must be unique")
            
            # Check for missing values first (before checking binary values)
            if df.isnull().any().any():
                raise ValueError("Submission file cannot contain missing values")
            
            # Check Survived values are binary
            if not pd.api.types.is_numeric_dtype(df['Survived']):
                raise ValueError("Survived column must be numeric")
            
            valid_survived_values = df['Survived'].isin([0, 1])
            if not valid_survived_values.all():
                raise ValueError("Survived column must contain only 0 or 1 values")
            
            logger.info("Submission file format validation passed")
            logger.info(f"Submission summary: {len(df)} rows, {df['Survived'].sum()} survivors")
            
            return True
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"Submission file is empty: {filepath}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing submission file: {e}")
        except Exception as e:
            logger.error(f"Submission validation failed: {str(e)}")
            raise
    
    def validate_competition_compliance(self, filepath: str) -> dict:
        """
        Comprehensive validation against Titanic competition requirements.
        
        Args:
            filepath: Path to submission file
            
        Returns:
            dict: Detailed validation report
            
        Raises:
            ValueError: If critical validation fails
            FileNotFoundError: If file doesn't exist
        """
        logger.info(f"Performing comprehensive competition compliance check: {filepath}")
        
        validation_report = {
            'file_exists': False,
            'correct_format': False,
            'correct_columns': False,
            'correct_row_count': False,
            'binary_predictions': False,
            'no_missing_values': False,
            'unique_passenger_ids': False,
            'csv_format': False,
            'competition_compliant': False,
            'details': {}
        }
        
        try:
            # Check file exists
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Submission file not found: {filepath}")
            validation_report['file_exists'] = True
            
            # Check if it's a CSV file
            if not filepath.lower().endswith('.csv'):
                validation_report['details']['format_error'] = "File must have .csv extension"
            else:
                validation_report['csv_format'] = True
            
            # Load and validate content
            df = pd.read_csv(filepath)
            validation_report['correct_format'] = True
            
            # Check columns
            expected_columns = ['PassengerId', 'Survived']
            if list(df.columns) == expected_columns:
                validation_report['correct_columns'] = True
            else:
                validation_report['details']['column_error'] = f"Expected {expected_columns}, found {list(df.columns)}"
            
            # Check row count (exactly 418 + header)
            if len(df) == 418:
                validation_report['correct_row_count'] = True
            else:
                validation_report['details']['row_count_error'] = f"Expected 418 rows, found {len(df)}"
            
            # Check for missing values
            if not df.isnull().any().any():
                validation_report['no_missing_values'] = True
            else:
                missing_info = df.isnull().sum().to_dict()
                validation_report['details']['missing_values'] = missing_info
            
            # Check PassengerId uniqueness
            if not df['PassengerId'].duplicated().any():
                validation_report['unique_passenger_ids'] = True
            else:
                duplicates = df[df['PassengerId'].duplicated()]['PassengerId'].tolist()
                validation_report['details']['duplicate_ids'] = duplicates
            
            # Check binary predictions
            if df['Survived'].isin([0, 1]).all():
                validation_report['binary_predictions'] = True
                validation_report['details']['prediction_summary'] = {
                    'total': len(df),
                    'survivors': int(df['Survived'].sum()),
                    'non_survivors': int((df['Survived'] == 0).sum()),
                    'survival_rate': float(df['Survived'].mean())
                }
            else:
                invalid_values = df[~df['Survived'].isin([0, 1])]['Survived'].unique().tolist()
                validation_report['details']['invalid_predictions'] = invalid_values
            
            # Overall compliance check
            compliance_checks = [
                validation_report['file_exists'],
                validation_report['correct_format'],
                validation_report['correct_columns'],
                validation_report['correct_row_count'],
                validation_report['binary_predictions'],
                validation_report['no_missing_values'],
                validation_report['unique_passenger_ids'],
                validation_report['csv_format']
            ]
            
            validation_report['competition_compliant'] = all(compliance_checks)
            
            if validation_report['competition_compliant']:
                logger.info("✅ Submission file is fully competition compliant")
            else:
                failed_checks = []
                check_names = [
                    'file_exists', 'correct_format', 'correct_columns', 
                    'correct_row_count', 'binary_predictions', 'no_missing_values',
                    'unique_passenger_ids', 'csv_format'
                ]
                for i, check in enumerate(compliance_checks):
                    if not check:
                        failed_checks.append(check_names[i])
                logger.warning(f"❌ Submission file failed checks: {failed_checks}")
            
            return validation_report
            
        except FileNotFoundError:
            # Re-raise FileNotFoundError as specified in docstring
            raise
        except pd.errors.EmptyDataError:
            validation_report['details']['error'] = "Submission file is empty"
            return validation_report
        except pd.errors.ParserError as e:
            validation_report['details']['error'] = f"Error parsing CSV file: {e}"
            return validation_report
        except Exception as e:
            validation_report['details']['error'] = str(e)
            return validation_report
    
    def predict_and_submit(self, X_test: pd.DataFrame, passenger_ids: pd.Series,
                          output_path: str = "outputs/predictions/submission.csv") -> Tuple[np.ndarray, str]:
        """
        Generate predictions and create submission file in one step.
        
        Args:
            X_test: Preprocessed test features
            passenger_ids: Passenger IDs for submission
            output_path: Path for submission file
            
        Returns:
            Tuple of (predictions array, submission file path)
        """
        logger.info("Starting prediction and submission generation")
        
        # Generate predictions
        predictions = self.generate_predictions(X_test, passenger_ids)
        
        # Create submission file
        submission_path = self.create_submission_file(predictions, passenger_ids, output_path)
        
        logger.info("Prediction and submission generation completed")
        
        return predictions, submission_path
    
    def get_prediction_summary(self) -> dict:
        """
        Get summary statistics of generated predictions.
        
        Returns:
            dict: Summary statistics
            
        Raises:
            ValueError: If no predictions are available
        """
        if self.predictions is None:
            raise ValueError("No predictions available. Generate predictions first.")
        
        total_predictions = len(self.predictions)
        survivors = np.sum(self.predictions == 1)
        non_survivors = np.sum(self.predictions == 0)
        survival_rate = survivors / total_predictions
        
        summary = {
            'total_predictions': total_predictions,
            'survivors': int(survivors),
            'non_survivors': int(non_survivors),
            'survival_rate': survival_rate,
            'prediction_distribution': {
                'died': int(non_survivors),
                'survived': int(survivors)
            }
        }
        
        return summary