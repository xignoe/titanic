"""
Advanced data preprocessing module for the Titanic survival predictor.

This module implements sophisticated feature engineering and preprocessing
techniques to maximize model performance.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
import logging
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AdvancedDataPreprocessor:
    """
    Advanced data preprocessor with sophisticated feature engineering.
    """
    
    def __init__(self):
        """Initialize the AdvancedDataPreprocessor."""
        self.age_imputation_values = {}
        self.fare_imputation_values = {}
        self.embarked_mode = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.is_fitted = False
        self.selected_features = None
        
    def fit(self, df: pd.DataFrame) -> 'AdvancedDataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Training dataframe
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting advanced preprocessor on training data")
        
        # Calculate sophisticated age imputation values
        self._fit_age_imputation(df)
        
        # Calculate fare imputation values
        self._fit_fare_imputation(df)
        
        # Get most frequent embarkation port
        self.embarked_mode = df['Embarked'].mode().iloc[0] if not df['Embarked'].mode().empty else 'S'
        
        self.is_fitted = True
        logger.info("Advanced preprocessor fitted successfully")
        return self
    
    def _fit_age_imputation(self, df: pd.DataFrame) -> None:
        """Fit sophisticated age imputation model."""
        # Create age groups based on title, class, and family structure
        df_temp = df.copy()
        df_temp['Title'] = df_temp['Name'].apply(self._extract_title_from_name)
        df_temp['FamilySize'] = df_temp['SibSp'] + df_temp['Parch'] + 1
        df_temp['IsAlone'] = (df_temp['FamilySize'] == 1).astype(int)
        
        # Multi-level imputation strategy
        for title in df_temp['Title'].unique():
            for pclass in df_temp['Pclass'].unique():
                for sex in df_temp['Sex'].unique():
                    mask = (df_temp['Title'] == title) & (df_temp['Pclass'] == pclass) & (df_temp['Sex'] == sex)
                    age_values = df_temp.loc[mask, 'Age'].dropna()
                    if len(age_values) > 0:
                        self.age_imputation_values[(title, pclass, sex)] = age_values.median()
        
        # Fallback strategies
        for pclass in df_temp['Pclass'].unique():
            for sex in df_temp['Sex'].unique():
                mask = (df_temp['Pclass'] == pclass) & (df_temp['Sex'] == sex)
                age_values = df_temp.loc[mask, 'Age'].dropna()
                if len(age_values) > 0:
                    self.age_imputation_values[('fallback', pclass, sex)] = age_values.median()
    
    def _fit_fare_imputation(self, df: pd.DataFrame) -> None:
        """Fit sophisticated fare imputation model."""
        # Multi-level fare imputation
        for pclass in df['Pclass'].unique():
            for embarked in df['Embarked'].dropna().unique():
                mask = (df['Pclass'] == pclass) & (df['Embarked'] == embarked)
                fare_values = df.loc[mask, 'Fare'].dropna()
                if len(fare_values) > 0:
                    self.fare_imputation_values[(pclass, embarked)] = fare_values.median()
        
        # Fallback by class only
        for pclass in df['Pclass'].unique():
            fare_values = df[df['Pclass'] == pclass]['Fare'].dropna()
            if len(fare_values) > 0:
                self.fare_imputation_values[('fallback', pclass)] = fare_values.median()
    
    def _extract_title_from_name(self, name: str) -> str:
        """Extract and normalize title from passenger name."""
        title_search = re.search(r' ([A-Za-z]+)\.', name)
        if title_search:
            title = title_search.group(1)
            # More sophisticated title grouping
            if title in ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
                return 'Rare'
            elif title in ['Mlle', 'Ms']:
                return 'Miss'
            elif title == 'Mme':
                return 'Mrs'
            elif title in ['Master']:
                return 'Master'
            elif title in ['Mr']:
                return 'Mr'
            elif title in ['Mrs']:
                return 'Mrs'
            elif title in ['Miss']:
                return 'Miss'
            else:
                return 'Other'
        return 'Unknown'
    
    def _extract_deck_from_cabin(self, cabin: str) -> str:
        """Extract deck information from cabin data."""
        if pd.isna(cabin) or cabin == '':
            return 'Unknown'
        return cabin[0]
    
    def _extract_cabin_number(self, cabin: str) -> int:
        """Extract cabin number from cabin data."""
        if pd.isna(cabin) or cabin == '':
            return -1
        # Extract numbers from cabin string
        numbers = re.findall(r'\d+', cabin)
        if numbers:
            return int(numbers[0])
        return -1
    
    def _extract_ticket_prefix(self, ticket: str) -> str:
        """Extract ticket prefix."""
        if pd.isna(ticket):
            return 'Unknown'
        # Extract non-numeric prefix
        prefix = re.sub(r'\d+', '', ticket).strip()
        if prefix == '':
            return 'Numeric'
        return prefix.replace('/', '').replace('.', '').strip()
    
    def _extract_ticket_number(self, ticket: str) -> int:
        """Extract ticket number."""
        if pd.isna(ticket):
            return -1
        numbers = re.findall(r'\d+', ticket)
        if numbers:
            return int(numbers[-1])  # Take the last number
        return -1
    
    def impute_age_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced age imputation using multiple strategies."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before imputing values")
        
        df_copy = df.copy()
        df_copy['Title'] = df_copy['Name'].apply(self._extract_title_from_name)
        missing_age_count = df_copy['Age'].isna().sum()
        
        if missing_age_count > 0:
            logger.info(f"Imputing {missing_age_count} missing age values using advanced strategy")
            
            for idx, row in df_copy[df_copy['Age'].isna()].iterrows():
                title, pclass, sex = row['Title'], row['Pclass'], row['Sex']
                
                # Try primary strategy
                if (title, pclass, sex) in self.age_imputation_values:
                    df_copy.loc[idx, 'Age'] = self.age_imputation_values[(title, pclass, sex)]
                # Try fallback strategy
                elif ('fallback', pclass, sex) in self.age_imputation_values:
                    df_copy.loc[idx, 'Age'] = self.age_imputation_values[('fallback', pclass, sex)]
                # Final fallback
                else:
                    df_copy.loc[idx, 'Age'] = df_copy['Age'].median()
        
        return df_copy
    
    def impute_fare_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced fare imputation."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before imputing values")
        
        df_copy = df.copy()
        missing_fare_count = df_copy['Fare'].isna().sum()
        
        if missing_fare_count > 0:
            logger.info(f"Imputing {missing_fare_count} missing fare values using advanced strategy")
            
            for idx, row in df_copy[df_copy['Fare'].isna()].iterrows():
                pclass, embarked = row['Pclass'], row['Embarked']
                
                # Try primary strategy
                if (pclass, embarked) in self.fare_imputation_values:
                    df_copy.loc[idx, 'Fare'] = self.fare_imputation_values[(pclass, embarked)]
                # Try fallback strategy
                elif ('fallback', pclass) in self.fare_imputation_values:
                    df_copy.loc[idx, 'Fare'] = self.fare_imputation_values[('fallback', pclass)]
                # Final fallback
                else:
                    df_copy.loc[idx, 'Fare'] = df_copy['Fare'].median()
        
        return df_copy
    
    def impute_embarked_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced embarked imputation."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before imputing values")
        
        df_copy = df.copy()
        missing_embarked_count = df_copy['Embarked'].isna().sum()
        
        if missing_embarked_count > 0:
            logger.info(f"Imputing {missing_embarked_count} missing embarked values")
            # For missing embarked, use fare and class to make educated guess
            for idx, row in df_copy[df_copy['Embarked'].isna()].iterrows():
                fare, pclass = row['Fare'], row['Pclass']
                
                # Calculate median fares by embarkation port for this class
                embarked_fares = {}
                for port in ['C', 'Q', 'S']:
                    port_fares = df_copy[(df_copy['Embarked'] == port) & (df_copy['Pclass'] == pclass)]['Fare']
                    if len(port_fares) > 0:
                        embarked_fares[port] = port_fares.median()
                
                # Choose the port with closest median fare
                if embarked_fares:
                    closest_port = min(embarked_fares.keys(), key=lambda x: abs(embarked_fares[x] - fare))
                    df_copy.loc[idx, 'Embarked'] = closest_port
                else:
                    df_copy.loc[idx, 'Embarked'] = self.embarked_mode
        
        return df_copy
    
    def engineer_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated derived features."""
        df_copy = df.copy()
        
        # Basic family features
        df_copy['FamilySize'] = df_copy['SibSp'] + df_copy['Parch'] + 1
        df_copy['IsAlone'] = (df_copy['FamilySize'] == 1).astype(int)
        
        # Advanced family features
        df_copy['HasSiblings'] = (df_copy['SibSp'] > 0).astype(int)
        df_copy['HasParentsChildren'] = (df_copy['Parch'] > 0).astype(int)
        df_copy['LargeFamilySize'] = (df_copy['FamilySize'] > 4).astype(int)
        df_copy['SmallFamilySize'] = (df_copy['FamilySize'] == 2).astype(int)
        
        # Title extraction
        df_copy['Title'] = df_copy['Name'].apply(self._extract_title_from_name)
        
        # Cabin features
        df_copy['Deck'] = df_copy['Cabin'].apply(self._extract_deck_from_cabin)
        df_copy['CabinNumber'] = df_copy['Cabin'].apply(self._extract_cabin_number)
        df_copy['HasCabin'] = (~df_copy['Cabin'].isna()).astype(int)
        cabin_numbers = df_copy['CabinNumber'].replace(-1, np.nan)
        df_copy['CabinNumberBand'] = pd.cut(cabin_numbers, bins=5, labels=['Low', 'Med-Low', 'Med', 'Med-High', 'High'])
        df_copy['CabinNumberBand'] = df_copy['CabinNumberBand'].astype(str).fillna('Unknown')
        
        # Ticket features
        df_copy['TicketPrefix'] = df_copy['Ticket'].apply(self._extract_ticket_prefix)
        df_copy['TicketNumber'] = df_copy['Ticket'].apply(self._extract_ticket_number)
        df_copy['TicketLength'] = df_copy['Ticket'].astype(str).str.len()
        
        # Fare features
        df_copy['FarePerPerson'] = df_copy['Fare'] / df_copy['FamilySize']
        df_copy['FareBand'] = pd.cut(df_copy['Fare'], bins=5, labels=['Low', 'Med-Low', 'Med', 'Med-High', 'High'])
        df_copy['FareBand'] = df_copy['FareBand'].astype(str).fillna('Unknown')
        df_copy['ExpensiveTicket'] = (df_copy['Fare'] > df_copy['Fare'].quantile(0.8)).astype(int)
        df_copy['CheapTicket'] = (df_copy['Fare'] < df_copy['Fare'].quantile(0.2)).astype(int)
        
        # Age features
        df_copy['AgeBand'] = pd.cut(df_copy['Age'], bins=6, labels=['Child', 'Teen', 'Young', 'Adult', 'Middle', 'Senior'])
        df_copy['AgeBand'] = df_copy['AgeBand'].astype(str).fillna('Unknown')
        df_copy['IsChild'] = (df_copy['Age'] < 16).astype(int)
        df_copy['IsElderly'] = (df_copy['Age'] > 60).astype(int)
        df_copy['AgeGroup'] = df_copy['Age'].apply(self._categorize_age)
        
        # Interaction features
        df_copy['Sex_Pclass'] = df_copy['Sex'].astype(str) + '_' + df_copy['Pclass'].astype(str)
        df_copy['Title_Pclass'] = df_copy['Title'].astype(str) + '_' + df_copy['Pclass'].astype(str)
        df_copy['Embarked_Pclass'] = df_copy['Embarked'].astype(str) + '_' + df_copy['Pclass'].astype(str)
        df_copy['FamilySize_Pclass'] = df_copy['FamilySize'].astype(str) + '_' + df_copy['Pclass'].astype(str)
        
        # Name length (might indicate social status)
        df_copy['NameLength'] = df_copy['Name'].str.len()
        
        logger.info("Created advanced engineered features")
        return df_copy
    
    def _categorize_age(self, age: float) -> str:
        """Categorize age into meaningful groups."""
        if pd.isna(age):
            return 'Unknown'
        elif age < 5:
            return 'Baby'
        elif age < 12:
            return 'Child'
        elif age < 18:
            return 'Teen'
        elif age < 25:
            return 'YoungAdult'
        elif age < 35:
            return 'Adult'
        elif age < 50:
            return 'MiddleAge'
        elif age < 65:
            return 'Senior'
        else:
            return 'Elderly'
    
    def encode_categorical_advanced(self, df: pd.DataFrame, fit_encoders: bool = False) -> pd.DataFrame:
        """Advanced categorical encoding with multiple strategies."""
        df_copy = df.copy()
        
        # One-hot encode basic categorical variables
        categorical_cols = ['Sex', 'Embarked', 'Title', 'Deck', 'AgeBand', 'FareBand', 
                           'AgeGroup', 'CabinNumberBand', 'TicketPrefix']
        
        for col in categorical_cols:
            if col in df_copy.columns:
                dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=False)
                df_copy = pd.concat([df_copy, dummies], axis=1)
        
        # Label encode interaction features (high cardinality)
        interaction_cols = ['Sex_Pclass', 'Title_Pclass', 'Embarked_Pclass', 'FamilySize_Pclass']
        
        for col in interaction_cols:
            if col in df_copy.columns:
                if fit_encoders:
                    le = LabelEncoder()
                    df_copy[f'{col}_encoded'] = le.fit_transform(df_copy[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        le = self.label_encoders[col]
                        df_copy[f'{col}_encoded'] = df_copy[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        df_copy[f'{col}_encoded'] = 0
        
        logger.info("Applied advanced categorical encoding")
        return df_copy
    
    def select_features(self, X: pd.DataFrame, y: pd.Series = None, 
                       fit_selector: bool = False, k: int = 50) -> pd.DataFrame:
        """Advanced feature selection."""
        if fit_selector and y is not None:
            # Use multiple feature selection methods
            selector_f = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
            selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
            
            X_f = selector_f.fit_transform(X, y)
            X_mi = selector_mi.fit_transform(X, y)
            
            # Get selected features from both methods
            selected_f = X.columns[selector_f.get_support()].tolist()
            selected_mi = X.columns[selector_mi.get_support()].tolist()
            
            # Combine and take union
            self.selected_features = list(set(selected_f + selected_mi))
            
            # Ensure we have important base features
            important_features = ['Pclass', 'Age', 'Fare', 'FamilySize', 'IsAlone']
            for feat in important_features:
                if feat in X.columns and feat not in self.selected_features:
                    self.selected_features.append(feat)
            
            logger.info(f"Selected {len(self.selected_features)} features using advanced selection")
            
        if self.selected_features:
            available_features = [f for f in self.selected_features if f in X.columns]
            return X[available_features]
        else:
            return X
    
    def preprocess_advanced(self, df: pd.DataFrame, target_col: str = 'Survived', 
                           fit_scaler: bool = False, fit_encoders: bool = False,
                           fit_selector: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """Complete advanced preprocessing pipeline."""
        logger.info("Starting advanced preprocessing pipeline")
        
        # Handle missing values with advanced strategies
        df_processed = self.impute_age_advanced(df)
        df_processed = self.impute_fare_advanced(df_processed)
        df_processed = self.impute_embarked_advanced(df_processed)
        
        # Engineer advanced features
        df_processed = self.engineer_advanced_features(df_processed)
        
        # Advanced categorical encoding
        df_processed = self.encode_categorical_advanced(df_processed, fit_encoders=fit_encoders)
        
        # Select numerical and encoded categorical features
        feature_cols = []
        
        # Base numerical features
        numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone',
                             'HasSiblings', 'HasParentsChildren', 'LargeFamilySize', 'SmallFamilySize',
                             'HasCabin', 'TicketNumber', 'TicketLength', 'FarePerPerson', 
                             'ExpensiveTicket', 'CheapTicket', 'IsChild', 'IsElderly', 'NameLength',
                             'CabinNumber']
        
        for col in numerical_features:
            if col in df_processed.columns:
                feature_cols.append(col)
        
        # Add one-hot encoded features
        for col in df_processed.columns:
            if any(col.startswith(prefix + '_') for prefix in ['Sex', 'Embarked', 'Title', 'Deck', 
                                                              'AgeBand', 'FareBand', 'AgeGroup', 
                                                              'CabinNumberBand', 'TicketPrefix']):
                feature_cols.append(col)
        
        # Add label encoded interaction features
        for col in df_processed.columns:
            if col.endswith('_encoded'):
                feature_cols.append(col)
        
        # Select available features
        available_features = [col for col in feature_cols if col in df_processed.columns]
        X = df_processed[available_features]
        
        # Handle any remaining missing values and ensure all columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                # Convert any remaining object columns to numeric using label encoding
                if col not in self.label_encoders:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    X.loc[:, col] = le.fit_transform(X[col].astype(str))
                    if fit_encoders:
                        self.label_encoders[col] = le
                else:
                    le = self.label_encoders[col]
                    X.loc[:, col] = X[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        X = X.fillna(0)
        
        # Feature selection
        y = df_processed[target_col] if target_col in df_processed.columns else None
        if y is not None:
            X = self.select_features(X, y, fit_selector=fit_selector)
        elif self.selected_features:
            available_selected = [f for f in self.selected_features if f in X.columns]
            X = X[available_selected]
        
        # Scale numerical features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            logger.info("Fitted scaler and scaled features")
        else:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            logger.info("Scaled features using fitted scaler")
        
        logger.info(f"Advanced preprocessing completed. Final shape: {X.shape}")
        return X, y