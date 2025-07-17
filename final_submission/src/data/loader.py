"""
Data loading utilities for the Titanic Survival Predictor.

This module provides the DataLoader class for loading and validating
the Titanic dataset files (train.csv and test.csv).
"""

import pandas as pd
import os
from typing import Optional, List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and validation of Titanic dataset files.
    
    This class provides methods to load training and test datasets,
    validate their format, and handle common data loading errors.
    """
    
    # Expected columns for training data
    TRAIN_REQUIRED_COLUMNS = [
        'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
        'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
    ]
    
    # Expected columns for test data (no 'Survived' column)
    TEST_REQUIRED_COLUMNS = [
        'PassengerId', 'Pclass', 'Name', 'Sex', 'Age',
        'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
    ]
    
    def __init__(self, data_dir: str = "titanic"):
        """
        Initialize DataLoader with data directory path.
        
        Args:
            data_dir: Directory containing the CSV files
        """
        self.data_dir = data_dir
        self.train_file = os.path.join(data_dir, "train.csv")
        self.test_file = os.path.join(data_dir, "test.csv")
    
    def load_train_data(self) -> pd.DataFrame:
        """
        Load the training dataset with survival outcomes.
        
        Returns:
            pandas.DataFrame: Training data with all features and target variable
            
        Raises:
            FileNotFoundError: If train.csv file is not found
            ValueError: If data format is invalid
        """
        try:
            if not os.path.exists(self.train_file):
                raise FileNotFoundError(f"Training file not found: {self.train_file}")
            
            logger.info(f"Loading training data from {self.train_file}")
            df = pd.read_csv(self.train_file)
            
            # Validate data format
            self.validate_data_format(df, is_training=True)
            
            logger.info(f"Successfully loaded {len(df)} training records")
            return df
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"Training file is empty: {self.train_file}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing training file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading training data: {e}")
            raise
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load the test dataset for prediction.
        
        Returns:
            pandas.DataFrame: Test data with features (no target variable)
            
        Raises:
            FileNotFoundError: If test.csv file is not found
            ValueError: If data format is invalid
        """
        try:
            if not os.path.exists(self.test_file):
                raise FileNotFoundError(f"Test file not found: {self.test_file}")
            
            logger.info(f"Loading test data from {self.test_file}")
            df = pd.read_csv(self.test_file)
            
            # Validate data format
            self.validate_data_format(df, is_training=False)
            
            logger.info(f"Successfully loaded {len(df)} test records")
            return df
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"Test file is empty: {self.test_file}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing test file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading test data: {e}")
            raise
    
    def validate_data_format(self, df: pd.DataFrame, is_training: bool = True) -> bool:
        """
        Validate that the DataFrame has the expected format and columns.
        
        Args:
            df: DataFrame to validate
            is_training: Whether this is training data (includes 'Survived' column)
            
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Check required columns
        required_columns = self.TRAIN_REQUIRED_COLUMNS if is_training else self.TEST_REQUIRED_COLUMNS
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for unexpected extra columns (warn but don't fail)
        extra_columns = set(df.columns) - set(required_columns)
        if extra_columns:
            logger.warning(f"Found unexpected columns: {extra_columns}")
        
        # Validate data types and ranges
        self._validate_data_types(df, is_training)
        
        # Validate expected record counts
        if is_training and len(df) != 891:
            logger.warning(f"Expected 891 training records, found {len(df)}")
        elif not is_training and len(df) != 418:
            logger.warning(f"Expected 418 test records, found {len(df)}")
        
        logger.info("Data format validation passed")
        return True
    
    def _validate_data_types(self, df: pd.DataFrame, is_training: bool) -> None:
        """
        Validate data types and value ranges for key columns.
        
        Args:
            df: DataFrame to validate
            is_training: Whether this is training data
            
        Raises:
            ValueError: If data type validation fails
        """
        # Check PassengerId is numeric and unique
        if not pd.api.types.is_numeric_dtype(df['PassengerId']):
            raise ValueError("PassengerId must be numeric")
        
        if df['PassengerId'].duplicated().any():
            raise ValueError("PassengerId values must be unique")
        
        # Check Survived column for training data
        if is_training:
            if not pd.api.types.is_numeric_dtype(df['Survived']):
                raise ValueError("Survived column must be numeric")
            
            valid_survival_values = df['Survived'].dropna().isin([0, 1])
            if not valid_survival_values.all():
                raise ValueError("Survived column must contain only 0 or 1 values")
        
        # Check Pclass values
        if not pd.api.types.is_numeric_dtype(df['Pclass']):
            raise ValueError("Pclass must be numeric")
        
        valid_pclass_values = df['Pclass'].dropna().isin([1, 2, 3])
        if not valid_pclass_values.all():
            raise ValueError("Pclass must contain only values 1, 2, or 3")
        
        # Check Sex values
        valid_sex_values = df['Sex'].dropna().isin(['male', 'female'])
        if not valid_sex_values.all():
            raise ValueError("Sex column must contain only 'male' or 'female' values")
        
        # Check numeric columns are actually numeric (allowing NaN)
        numeric_columns = ['Age', 'SibSp', 'Parch', 'Fare']
        for col in numeric_columns:
            if col in df.columns:
                # Convert to numeric, errors='coerce' will turn invalid values to NaN
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.isna().sum() > df[col].isna().sum():
                    raise ValueError(f"Column {col} contains non-numeric values")
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get basic information about the loaded dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            dict: Information about the dataset
        """
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }