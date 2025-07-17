"""
Unit tests for the DataPreprocessor class.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5, 6, 7, 8],
            'Pclass': [1, 1, 2, 2, 3, 3, 1, 2],
            'Sex': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
            'Age': [22.0, 38.0, np.nan, 35.0, np.nan, np.nan, 54.0, 27.0],
            'SibSp': [1, 0, 2, 1, 0, 1, 0, 2],
            'Parch': [0, 1, 1, 0, 2, 0, 1, 0],
            'Fare': [7.25, 71.28, np.nan, 53.10, 8.05, np.nan, 51.86, 21.08],
            'Embarked': ['S', 'C', np.nan, 'S', 'S', 'Q', 'S', np.nan]
        })
    
    @pytest.fixture
    def preprocessor(self, sample_data):
        """Create a fitted preprocessor."""
        processor = DataPreprocessor()
        processor.fit(sample_data)
        return processor
    
    def test_init(self):
        """Test DataPreprocessor initialization."""
        processor = DataPreprocessor()
        assert processor.age_imputation_values == {}
        assert processor.fare_imputation_values == {}
        assert processor.embarked_mode is None
        assert not processor.is_fitted
    
    def test_fit(self, sample_data):
        """Test fitting the preprocessor."""
        processor = DataPreprocessor()
        result = processor.fit(sample_data)
        
        # Should return self for method chaining
        assert result is processor
        assert processor.is_fitted
        
        # Check age imputation values are calculated correctly
        assert len(processor.age_imputation_values) > 0
        assert (1, 'male') in processor.age_imputation_values
        assert (1, 'female') in processor.age_imputation_values
        
        # Check fare imputation values are calculated
        assert len(processor.fare_imputation_values) > 0
        
        # Check embarked mode is set
        assert processor.embarked_mode in ['S', 'C', 'Q']
    
    def test_impute_age_not_fitted(self, sample_data):
        """Test that impute_age raises error when not fitted."""
        processor = DataPreprocessor()
        with pytest.raises(ValueError, match="Preprocessor must be fitted"):
            processor.impute_age(sample_data)
    
    def test_impute_age(self, preprocessor, sample_data):
        """Test age imputation."""
        result = preprocessor.impute_age(sample_data)
        
        # Should not modify original dataframe
        assert sample_data['Age'].isna().sum() > 0
        
        # Result should have no missing age values
        assert result['Age'].isna().sum() == 0
        
        # Non-missing values should remain unchanged
        assert result.loc[0, 'Age'] == 22.0
        assert result.loc[1, 'Age'] == 38.0
        assert result.loc[3, 'Age'] == 35.0
        
        # Missing values should be imputed with reasonable values
        assert result.loc[2, 'Age'] > 0  # Age should be positive
        assert result.loc[4, 'Age'] > 0
        assert result.loc[5, 'Age'] > 0
    
    def test_impute_fare_not_fitted(self, sample_data):
        """Test that impute_fare raises error when not fitted."""
        processor = DataPreprocessor()
        with pytest.raises(ValueError, match="Preprocessor must be fitted"):
            processor.impute_fare(sample_data)
    
    def test_impute_fare(self, preprocessor, sample_data):
        """Test fare imputation."""
        result = preprocessor.impute_fare(sample_data)
        
        # Should not modify original dataframe
        assert sample_data['Fare'].isna().sum() > 0
        
        # Result should have no missing fare values
        assert result['Fare'].isna().sum() == 0
        
        # Non-missing values should remain unchanged
        assert result.loc[0, 'Fare'] == 7.25
        assert result.loc[1, 'Fare'] == 71.28
        assert result.loc[3, 'Fare'] == 53.10
        
        # Missing values should be imputed with reasonable values
        assert result.loc[2, 'Fare'] >= 0  # Fare should be non-negative
        assert result.loc[5, 'Fare'] >= 0
    
    def test_impute_embarked_not_fitted(self, sample_data):
        """Test that impute_embarked raises error when not fitted."""
        processor = DataPreprocessor()
        with pytest.raises(ValueError, match="Preprocessor must be fitted"):
            processor.impute_embarked(sample_data)
    
    def test_impute_embarked(self, preprocessor, sample_data):
        """Test embarked imputation."""
        result = preprocessor.impute_embarked(sample_data)
        
        # Should not modify original dataframe
        assert sample_data['Embarked'].isna().sum() > 0
        
        # Result should have no missing embarked values
        assert result['Embarked'].isna().sum() == 0
        
        # Non-missing values should remain unchanged
        assert result.loc[0, 'Embarked'] == 'S'
        assert result.loc[1, 'Embarked'] == 'C'
        assert result.loc[3, 'Embarked'] == 'S'
        
        # Missing values should be imputed with valid port codes
        assert result.loc[2, 'Embarked'] in ['S', 'C', 'Q']
        assert result.loc[7, 'Embarked'] in ['S', 'C', 'Q']
    
    def test_handle_missing_values(self, preprocessor, sample_data):
        """Test comprehensive missing value handling."""
        result = preprocessor.handle_missing_values(sample_data)
        
        # Should not modify original dataframe
        original_missing = sample_data.isna().sum().sum()
        assert original_missing > 0
        
        # Result should have no missing values in key columns
        assert result['Age'].isna().sum() == 0
        assert result['Fare'].isna().sum() == 0
        assert result['Embarked'].isna().sum() == 0
        
        # Should preserve non-missing values
        assert result.loc[0, 'Age'] == 22.0
        assert result.loc[1, 'Fare'] == 71.28
        assert result.loc[0, 'Embarked'] == 'S'
    
    def test_age_imputation_by_group(self):
        """Test that age imputation uses correct group medians."""
        # Create data where group medians are clearly different
        data = pd.DataFrame({
            'Pclass': [1, 1, 1, 2, 2, 2],
            'Sex': ['male', 'male', 'male', 'female', 'female', 'female'],
            'Age': [30.0, 40.0, np.nan, 20.0, 25.0, np.nan],
            'Fare': [50.0, 60.0, 55.0, 30.0, 35.0, 32.0],
            'Embarked': ['S', 'S', 'S', 'C', 'C', 'C']
        })
        
        processor = DataPreprocessor()
        processor.fit(data)
        result = processor.impute_age(data)
        
        # Class 1 male median should be 35.0 (median of 30, 40)
        assert result.loc[2, 'Age'] == 35.0
        
        # Class 2 female median should be 22.5 (median of 20, 25)
        assert result.loc[5, 'Age'] == 22.5
    
    def test_fare_imputation_by_group(self):
        """Test that fare imputation uses correct group medians."""
        # Create data where group medians are clearly different
        data = pd.DataFrame({
            'Pclass': [1, 1, 1, 2, 2, 2],
            'Sex': ['male', 'female', 'male', 'female', 'male', 'female'],
            'Age': [30.0, 40.0, 35.0, 20.0, 25.0, 22.0],
            'Fare': [100.0, 120.0, np.nan, 30.0, 40.0, np.nan],
            'Embarked': ['S', 'S', 'S', 'C', 'C', 'C']
        })
        
        processor = DataPreprocessor()
        processor.fit(data)
        result = processor.impute_fare(data)
        
        # Class 1, Embarked S median should be 110.0 (median of 100, 120)
        assert result.loc[2, 'Fare'] == 110.0
        
        # Class 2, Embarked C median should be 35.0 (median of 30, 40)
        assert result.loc[5, 'Fare'] == 35.0
    
    def test_embarked_mode_selection(self):
        """Test that embarked imputation uses the most frequent value."""
        data = pd.DataFrame({
            'Pclass': [1, 2, 3, 1, 2],
            'Sex': ['male', 'female', 'male', 'female', 'male'],
            'Age': [30.0, 40.0, 35.0, 20.0, 25.0],
            'Fare': [100.0, 120.0, 50.0, 30.0, 40.0],
            'Embarked': ['S', 'S', 'S', 'C', np.nan]  # S appears 3 times, C appears 1 time
        })
        
        processor = DataPreprocessor()
        processor.fit(data)
        result = processor.impute_embarked(data)
        
        # Missing value should be imputed with 'S' (most frequent)
        assert result.loc[4, 'Embarked'] == 'S'
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframes."""
        empty_df = pd.DataFrame(columns=['Pclass', 'Sex', 'Age', 'Fare', 'Embarked'])
        processor = DataPreprocessor()
        
        # Should handle empty dataframe gracefully
        processor.fit(empty_df)
        result = processor.handle_missing_values(empty_df)
        assert len(result) == 0
    
    def test_no_missing_values(self):
        """Test handling of data with no missing values."""
        complete_data = pd.DataFrame({
            'Pclass': [1, 2, 3],
            'Sex': ['male', 'female', 'male'],
            'Age': [30.0, 40.0, 35.0],
            'Fare': [100.0, 50.0, 25.0],
            'Embarked': ['S', 'C', 'Q']
        })
        
        processor = DataPreprocessor()
        processor.fit(complete_data)
        result = processor.handle_missing_values(complete_data)
        
        # Should return identical dataframe
        pd.testing.assert_frame_equal(result, complete_data)  
  
    def test_encode_categorical_features(self, preprocessor):
        """Test categorical feature encoding."""
        data = pd.DataFrame({
            'Sex': ['male', 'female', 'male'],
            'Embarked': ['S', 'C', 'Q']
        })
        
        result = preprocessor.encode_categorical_features(data)
        
        # Check Sex encoding
        assert 'Sex_male' in result.columns
        assert 'Sex_female' in result.columns
        assert result.loc[0, 'Sex_male'] == 1
        assert result.loc[0, 'Sex_female'] == 0
        assert result.loc[1, 'Sex_male'] == 0
        assert result.loc[1, 'Sex_female'] == 1
        
        # Check Embarked encoding
        assert 'Embarked_S' in result.columns
        assert 'Embarked_C' in result.columns
        assert 'Embarked_Q' in result.columns
        assert result.loc[0, 'Embarked_S'] == 1
        assert result.loc[1, 'Embarked_C'] == 1
        assert result.loc[2, 'Embarked_Q'] == 1
    
    def test_extract_title_from_name(self, preprocessor):
        """Test title extraction from names."""
        # Test common titles
        assert preprocessor.extract_title_from_name("Smith, Mr. John") == "Mr"
        assert preprocessor.extract_title_from_name("Johnson, Mrs. Mary") == "Mrs"
        assert preprocessor.extract_title_from_name("Brown, Miss. Sarah") == "Miss"
        assert preprocessor.extract_title_from_name("Wilson, Master. Tom") == "Master"
        
        # Test rare titles
        assert preprocessor.extract_title_from_name("Jones, Dr. Robert") == "Rare"
        assert preprocessor.extract_title_from_name("Davis, Col. James") == "Rare"
        
        # Test special cases
        assert preprocessor.extract_title_from_name("Miller, Mlle. Anne") == "Miss"
        assert preprocessor.extract_title_from_name("Taylor, Mme. Claire") == "Mrs"
        
        # Test no title
        assert preprocessor.extract_title_from_name("Anderson, John") == "Unknown"
    
    def test_extract_deck_from_cabin(self, preprocessor):
        """Test deck extraction from cabin data."""
        # Test valid cabin numbers
        assert preprocessor.extract_deck_from_cabin("C85") == "C"
        assert preprocessor.extract_deck_from_cabin("A36") == "A"
        assert preprocessor.extract_deck_from_cabin("B57 B59 B63 B66") == "B"
        
        # Test missing cabin data
        assert preprocessor.extract_deck_from_cabin(np.nan) == "Unknown"
        assert preprocessor.extract_deck_from_cabin("") == "Unknown"
    
    def test_engineer_features(self, preprocessor):
        """Test feature engineering."""
        data = pd.DataFrame({
            'SibSp': [1, 0, 2],
            'Parch': [0, 1, 1],
            'Name': ['Smith, Mr. John', 'Johnson, Mrs. Mary', 'Brown, Miss. Sarah'],
            'Cabin': ['C85', np.nan, 'A36'],
            'Fare': [30.0, 50.0, 20.0]
        })
        
        result = preprocessor.engineer_features(data)
        
        # Check FamilySize
        assert 'FamilySize' in result.columns
        assert result.loc[0, 'FamilySize'] == 2  # 1 + 0 + 1
        assert result.loc[1, 'FamilySize'] == 2  # 0 + 1 + 1
        assert result.loc[2, 'FamilySize'] == 4  # 2 + 1 + 1
        
        # Check IsAlone
        assert 'IsAlone' in result.columns
        assert result.loc[0, 'IsAlone'] == 0  # FamilySize = 2
        assert result.loc[1, 'IsAlone'] == 0  # FamilySize = 2
        assert result.loc[2, 'IsAlone'] == 0  # FamilySize = 4
        
        # Check Title
        assert 'Title' in result.columns
        assert result.loc[0, 'Title'] == 'Mr'
        assert result.loc[1, 'Title'] == 'Mrs'
        assert result.loc[2, 'Title'] == 'Miss'
        
        # Check Deck
        assert 'Deck' in result.columns
        assert result.loc[0, 'Deck'] == 'C'
        assert result.loc[1, 'Deck'] == 'Unknown'
        assert result.loc[2, 'Deck'] == 'A'
        
        # Check FarePerPerson
        assert 'FarePerPerson' in result.columns
        assert result.loc[0, 'FarePerPerson'] == 15.0  # 30.0 / 2
        assert result.loc[1, 'FarePerPerson'] == 25.0  # 50.0 / 2
        assert result.loc[2, 'FarePerPerson'] == 5.0   # 20.0 / 4
    
    def test_scale_numerical_features(self, preprocessor):
        """Test numerical feature scaling."""
        data = pd.DataFrame({
            'Age': [25.0, 35.0, 45.0],
            'Fare': [10.0, 50.0, 100.0],
            'FarePerPerson': [5.0, 25.0, 50.0],
            'Pclass': [1, 2, 3]  # Should not be scaled
        })
        
        # Test fitting scaler
        result = preprocessor.scale_numerical_features(data, fit_scaler=True)
        
        # Numerical columns should be scaled (mean ~0, std ~1)
        assert abs(result['Age'].mean()) < 0.1
        assert abs(result['Fare'].mean()) < 0.1
        assert abs(result['FarePerPerson'].mean()) < 0.1
        
        # Non-numerical columns should remain unchanged
        assert result['Pclass'].equals(data['Pclass'])
    
    def test_prepare_features(self, preprocessor):
        """Test feature preparation for modeling."""
        data = pd.DataFrame({
            'PassengerId': [1, 2, 3],
            'Survived': [0, 1, 0],
            'Pclass': [1, 2, 3],
            'Age': [25.0, 35.0, 45.0],
            'SibSp': [1, 0, 2],
            'Parch': [0, 1, 1],
            'Fare': [30.0, 50.0, 20.0],
            'Sex_male': [1, 0, 1],
            'Sex_female': [0, 1, 0],
            'Embarked_S': [1, 0, 1],
            'Embarked_C': [0, 1, 0],
            'Embarked_Q': [0, 0, 0],
            'FamilySize': [2, 2, 4],
            'IsAlone': [0, 0, 0],
            'FarePerPerson': [15.0, 25.0, 5.0],
            'Title': ['Mr', 'Mrs', 'Miss'],
            'Deck': ['C', 'Unknown', 'A']
        })
        
        X, y = preprocessor.prepare_features(data)
        
        # Check that target is extracted correctly
        assert y is not None
        assert len(y) == 3
        assert y.tolist() == [0, 1, 0]
        
        # Check that features are selected correctly
        assert 'PassengerId' not in X.columns
        assert 'Survived' not in X.columns
        assert 'Pclass' in X.columns
        assert 'Age' in X.columns
        
        # Check that Title and Deck are one-hot encoded
        title_cols = [col for col in X.columns if col.startswith('Title_')]
        deck_cols = [col for col in X.columns if col.startswith('Deck_')]
        assert len(title_cols) > 0
        assert len(deck_cols) > 0
    
    def test_preprocess_data_complete_pipeline(self, sample_data):
        """Test complete preprocessing pipeline."""
        # Add Name and Cabin columns for feature engineering
        sample_data['Name'] = [
            'Smith, Mr. John', 'Johnson, Mrs. Mary', 'Brown, Miss. Sarah',
            'Wilson, Master. Tom', 'Davis, Mr. Robert', 'Miller, Mrs. Anne',
            'Taylor, Dr. James', 'Anderson, Miss. Claire'
        ]
        sample_data['Cabin'] = ['C85', np.nan, 'A36', 'B57', np.nan, 'D20', 'E46', np.nan]
        sample_data['Survived'] = [0, 1, 0, 1, 0, 1, 1, 0]
        
        processor = DataPreprocessor()
        processor.fit(sample_data)
        
        X, y = processor.preprocess_data(sample_data, fit_scaler=True)
        
        # Check that we get features and target
        assert X is not None
        assert y is not None
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        
        # Check that no missing values remain in key features
        assert not X.isna().any().any()
        
        # Check that engineered features are present
        assert any(col.startswith('Title_') for col in X.columns)
        assert any(col.startswith('Deck_') for col in X.columns)
        assert 'FamilySize' in X.columns
        assert 'IsAlone' in X.columns
    
    def test_preprocess_test_data(self, sample_data):
        """Test preprocessing of test data (without target)."""
        # Prepare training data
        train_data = sample_data.copy()
        train_data['Name'] = [
            'Smith, Mr. John', 'Johnson, Mrs. Mary', 'Brown, Miss. Sarah',
            'Wilson, Master. Tom', 'Davis, Mr. Robert', 'Miller, Mrs. Anne',
            'Taylor, Dr. James', 'Anderson, Miss. Claire'
        ]
        train_data['Cabin'] = ['C85', np.nan, 'A36', 'B57', np.nan, 'D20', 'E46', np.nan]
        train_data['Survived'] = [0, 1, 0, 1, 0, 1, 1, 0]
        
        # Prepare test data (no Survived column)
        test_data = train_data.drop('Survived', axis=1)
        
        processor = DataPreprocessor()
        processor.fit(train_data)
        
        # Process training data first to fit scaler
        X_train, y_train = processor.preprocess_data(train_data, fit_scaler=True)
        
        # Process test data without fitting scaler
        X_test, y_test = processor.preprocess_data(test_data, target_col='Survived', fit_scaler=False)
        
        # Check that test processing works
        assert X_test is not None
        assert y_test is None  # No target in test data
        assert len(X_test) == len(test_data)
        
        # Check that feature columns match between train and test
        assert set(X_train.columns) == set(X_test.columns)
    
    def test_is_alone_feature(self, preprocessor):
        """Test IsAlone feature creation."""
        data = pd.DataFrame({
            'SibSp': [0, 1, 0, 2],
            'Parch': [0, 0, 1, 1]
        })
        
        result = preprocessor.engineer_features(data)
        
        # Check IsAlone logic
        assert result.loc[0, 'IsAlone'] == 1  # SibSp=0, Parch=0, FamilySize=1
        assert result.loc[1, 'IsAlone'] == 0  # SibSp=1, Parch=0, FamilySize=2
        assert result.loc[2, 'IsAlone'] == 0  # SibSp=0, Parch=1, FamilySize=2
        assert result.loc[3, 'IsAlone'] == 0  # SibSp=2, Parch=1, FamilySize=4