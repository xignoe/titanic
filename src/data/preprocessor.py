"""
Data preprocessing module for the Titanic survival predictor.

This module handles missing value imputation, categorical encoding,
and feature engineering for the Titanic dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
import logging
import re
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data preprocessing including missing value imputation,
    categorical encoding, and feature engineering.
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.age_imputation_values = {}
        self.fare_imputation_values = {}
        self.embarked_mode = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data to learn imputation values.
        
        Args:
            df: Training dataframe
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting preprocessor on training data")
        
        # Calculate age imputation values by passenger class and sex
        self.age_imputation_values = df.groupby(['Pclass', 'Sex'])['Age'].median().to_dict()
        
        # Calculate fare imputation values by passenger class and embarkation port
        fare_groups = df.groupby(['Pclass', 'Embarked'])['Fare'].median()
        self.fare_imputation_values = fare_groups.to_dict()
        
        # Get most frequent embarkation port
        self.embarked_mode = df['Embarked'].mode().iloc[0] if not df['Embarked'].mode().empty else 'S'
        
        self.is_fitted = True
        logger.info("Preprocessor fitted successfully")
        return self
    
    def impute_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing age values using median by passenger class and sex.
        
        Args:
            df: DataFrame with potential missing age values
            
        Returns:
            DataFrame with imputed age values
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before imputing values")
        
        df_copy = df.copy()
        missing_age_count = df_copy['Age'].isna().sum()
        
        if missing_age_count > 0:
            logger.info(f"Imputing {missing_age_count} missing age values")
            
            for (pclass, sex), median_age in self.age_imputation_values.items():
                mask = (df_copy['Pclass'] == pclass) & (df_copy['Sex'] == sex) & df_copy['Age'].isna()
                df_copy.loc[mask, 'Age'] = median_age
            
            # Handle any remaining missing values with overall median
            if df_copy['Age'].isna().any():
                overall_median = df_copy['Age'].median()
                df_copy.loc[:, 'Age'] = df_copy['Age'].fillna(overall_median)
                logger.warning("Some age values imputed with overall median due to missing group combinations")
        
        return df_copy
    
    def impute_fare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing fare values based on passenger class and embarkation port.
        
        Args:
            df: DataFrame with potential missing fare values
            
        Returns:
            DataFrame with imputed fare values
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before imputing values")
        
        df_copy = df.copy()
        missing_fare_count = df_copy['Fare'].isna().sum()
        
        if missing_fare_count > 0:
            logger.info(f"Imputing {missing_fare_count} missing fare values")
            
            for (pclass, embarked), median_fare in self.fare_imputation_values.items():
                mask = (df_copy['Pclass'] == pclass) & (df_copy['Embarked'] == embarked) & df_copy['Fare'].isna()
                df_copy.loc[mask, 'Fare'] = median_fare
            
            # Handle any remaining missing values with class median
            if df_copy['Fare'].isna().any():
                for pclass in df_copy['Pclass'].unique():
                    class_median = df_copy[df_copy['Pclass'] == pclass]['Fare'].median()
                    mask = (df_copy['Pclass'] == pclass) & df_copy['Fare'].isna()
                    df_copy.loc[mask, 'Fare'] = class_median
                logger.warning("Some fare values imputed with class median due to missing group combinations")
        
        return df_copy
    
    def impute_embarked(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing embarked values using the most frequent value.
        
        Args:
            df: DataFrame with potential missing embarked values
            
        Returns:
            DataFrame with imputed embarked values
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before imputing values")
        
        df_copy = df.copy()
        missing_embarked_count = df_copy['Embarked'].isna().sum()
        
        if missing_embarked_count > 0:
            logger.info(f"Imputing {missing_embarked_count} missing embarked values with '{self.embarked_mode}'")
            df_copy.loc[:, 'Embarked'] = df_copy['Embarked'].fillna(self.embarked_mode)
        
        return df_copy
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle all missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with all missing values imputed
        """
        logger.info("Handling missing values")
        
        # Apply all imputation methods
        df_processed = self.impute_age(df)
        df_processed = self.impute_fare(df_processed)
        df_processed = self.impute_embarked(df_processed)
        
        logger.info("Missing value imputation completed")
        return df_processed
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding.
        
        Args:
            df: DataFrame with categorical features
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_copy = df.copy()
        
        # One-hot encode Sex
        if 'Sex' in df_copy.columns:
            df_copy['Sex_male'] = (df_copy['Sex'] == 'male').astype(int)
            df_copy['Sex_female'] = (df_copy['Sex'] == 'female').astype(int)
            logger.info("Encoded Sex feature")
        
        # One-hot encode Embarked
        if 'Embarked' in df_copy.columns:
            df_copy['Embarked_C'] = (df_copy['Embarked'] == 'C').astype(int)
            df_copy['Embarked_Q'] = (df_copy['Embarked'] == 'Q').astype(int)
            df_copy['Embarked_S'] = (df_copy['Embarked'] == 'S').astype(int)
            logger.info("Encoded Embarked feature")
        
        return df_copy
    
    def extract_title_from_name(self, name: str) -> str:
        """
        Extract title from passenger name.
        
        Args:
            name: Passenger name string
            
        Returns:
            Extracted title
        """
        # Extract title using regex
        title_search = re.search(r' ([A-Za-z]+)\.', name)
        if title_search:
            title = title_search.group(1)
            # Group rare titles
            if title in ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']:
                return 'Rare'
            elif title in ['Mlle', 'Ms']:
                return 'Miss'
            elif title == 'Mme':
                return 'Mrs'
            else:
                return title
        return 'Unknown'
    
    def extract_deck_from_cabin(self, cabin: str) -> str:
        """
        Extract deck information from cabin data.
        
        Args:
            cabin: Cabin string
            
        Returns:
            Deck letter or 'Unknown'
        """
        if pd.isna(cabin) or cabin == '':
            return 'Unknown'
        # Extract first letter as deck
        return cabin[0]
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from existing data.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with engineered features
        """
        df_copy = df.copy()
        
        # Create FamilySize feature
        if 'SibSp' in df_copy.columns and 'Parch' in df_copy.columns:
            df_copy['FamilySize'] = df_copy['SibSp'] + df_copy['Parch'] + 1
            logger.info("Created FamilySize feature")
        
        # Create IsAlone feature
        if 'FamilySize' in df_copy.columns:
            df_copy['IsAlone'] = (df_copy['FamilySize'] == 1).astype(int)
            logger.info("Created IsAlone feature")
        
        # Extract Title from Name
        if 'Name' in df_copy.columns:
            df_copy['Title'] = df_copy['Name'].apply(self.extract_title_from_name)
            logger.info("Created Title feature")
        
        # Extract Deck from Cabin
        if 'Cabin' in df_copy.columns:
            df_copy['Deck'] = df_copy['Cabin'].apply(self.extract_deck_from_cabin)
            logger.info("Created Deck feature")
        
        # Create FarePerPerson feature
        if 'Fare' in df_copy.columns and 'FamilySize' in df_copy.columns:
            df_copy['FarePerPerson'] = df_copy['Fare'] / df_copy['FamilySize']
            # Handle division by zero (shouldn't happen with FamilySize >= 1)
            df_copy['FarePerPerson'] = df_copy['FarePerPerson'].fillna(df_copy['Fare'])
            logger.info("Created FarePerPerson feature")
        
        return df_copy
    
    def scale_numerical_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: DataFrame with numerical features
            fit_scaler: Whether to fit the scaler (True for training data)
            
        Returns:
            DataFrame with scaled numerical features
        """
        df_copy = df.copy()
        
        # Define numerical columns to scale
        numerical_cols = ['Age', 'Fare', 'FarePerPerson']
        available_cols = [col for col in numerical_cols if col in df_copy.columns]
        
        if available_cols:
            if fit_scaler:
                # Fit and transform for training data
                df_copy[available_cols] = self.scaler.fit_transform(df_copy[available_cols])
                logger.info(f"Fitted scaler and scaled features: {available_cols}")
            else:
                # Only transform for test data
                df_copy[available_cols] = self.scaler.transform(df_copy[available_cols])
                logger.info(f"Scaled features: {available_cols}")
        
        return df_copy
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'Survived', 
                        fit_scaler: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning by selecting relevant columns.
        
        Args:
            df: DataFrame with all features
            target_col: Name of target column
            fit_scaler: Whether to fit the scaler
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df_copy = df.copy()
        
        # Define base feature columns to keep for modeling
        feature_cols = [
            'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
            'Sex_male', 'Sex_female', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
            'FamilySize', 'IsAlone', 'FarePerPerson'
        ]
        
        # Handle Title features with alignment
        if 'Title' in df_copy.columns:
            # One-hot encode Title
            title_dummies = pd.get_dummies(df_copy['Title'], prefix='Title')
            df_copy = pd.concat([df_copy, title_dummies], axis=1)
            
            # If we're fitting (training), store the title columns
            if fit_scaler:
                self.title_columns = title_dummies.columns.tolist()
                feature_cols.extend(self.title_columns)
            else:
                # If we're transforming (test), align with training columns
                if hasattr(self, 'title_columns'):
                    # Add missing columns with zeros
                    for col in self.title_columns:
                        if col not in df_copy.columns:
                            df_copy[col] = 0
                    # Remove extra columns not seen in training
                    current_title_cols = [col for col in df_copy.columns if col.startswith('Title_')]
                    for col in current_title_cols:
                        if col not in self.title_columns:
                            df_copy = df_copy.drop(columns=[col])
                    feature_cols.extend(self.title_columns)
                else:
                    # Fallback if no stored columns
                    feature_cols.extend(title_dummies.columns.tolist())
        
        # Handle Deck features with alignment
        if 'Deck' in df_copy.columns:
            # One-hot encode Deck
            deck_dummies = pd.get_dummies(df_copy['Deck'], prefix='Deck')
            df_copy = pd.concat([df_copy, deck_dummies], axis=1)
            
            # If we're fitting (training), store the deck columns
            if fit_scaler:
                self.deck_columns = deck_dummies.columns.tolist()
                feature_cols.extend(self.deck_columns)
            else:
                # If we're transforming (test), align with training columns
                if hasattr(self, 'deck_columns'):
                    # Add missing columns with zeros
                    for col in self.deck_columns:
                        if col not in df_copy.columns:
                            df_copy[col] = 0
                    # Remove extra columns not seen in training
                    current_deck_cols = [col for col in df_copy.columns if col.startswith('Deck_')]
                    for col in current_deck_cols:
                        if col not in self.deck_columns:
                            df_copy = df_copy.drop(columns=[col])
                    feature_cols.extend(self.deck_columns)
                else:
                    # Fallback if no stored columns
                    feature_cols.extend(deck_dummies.columns.tolist())
        
        # Select only available feature columns
        available_features = [col for col in feature_cols if col in df_copy.columns]
        X = df_copy[available_features]
        
        # Get target variable if it exists
        y = df_copy[target_col] if target_col in df_copy.columns else None
        
        logger.info(f"Prepared {len(available_features)} features for modeling")
        return X, y
    
    def preprocess_data(self, df: pd.DataFrame, target_col: str = 'Survived', 
                       fit_scaler: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw DataFrame
            target_col: Name of target column
            fit_scaler: Whether to fit the scaler (True for training data)
            
        Returns:
            Tuple of (processed features DataFrame, target Series)
        """
        logger.info("Starting complete preprocessing pipeline")
        
        # Handle missing values
        df_processed = self.handle_missing_values(df)
        
        # Encode categorical features
        df_processed = self.encode_categorical_features(df_processed)
        
        # Engineer new features
        df_processed = self.engineer_features(df_processed)
        
        # Scale numerical features
        df_processed = self.scale_numerical_features(df_processed, fit_scaler=fit_scaler)
        
        # Prepare final feature set
        X, y = self.prepare_features(df_processed, target_col, fit_scaler=fit_scaler)
        
        logger.info("Preprocessing pipeline completed")
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names after preprocessing.
        
        Returns:
            List of feature names
        """
        # Base feature columns
        feature_cols = [
            'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
            'Sex_male', 'Sex_female', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
            'FamilySize', 'IsAlone', 'FarePerPerson'
        ]
        
        # Add actual title columns if available
        if hasattr(self, 'title_columns'):
            feature_cols.extend(self.title_columns)
        else:
            # Add common Title features as fallback
            title_features = ['Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare']
            feature_cols.extend(title_features)
        
        # Add actual deck columns if available
        if hasattr(self, 'deck_columns'):
            feature_cols.extend(self.deck_columns)
        else:
            # Add common Deck features as fallback
            deck_features = ['Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_T', 'Deck_Unknown']
            feature_cols.extend(deck_features)
        
        return feature_cols