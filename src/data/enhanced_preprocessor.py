import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def clean_transform_df(df, scaler=None, fit_scaler=True, feature_columns=None):
    """
    Enhanced preprocessing for Titanic Kaggle Competition with advanced feature engineering.
    
    Args:
        df: Input DataFrame from train.csv or test.csv
        scaler: Optional pre-fitted StandardScaler
        fit_scaler: Whether to fit the scaler (True for train, False for test)
        feature_columns: List of expected feature columns (for test set alignment)
        
    Returns:
        Processed feature DataFrame, scaler, and feature columns (if training)
    """
    df = df.copy()
    df = df.set_index('PassengerId')
    
    # ===== TITLE EXTRACTION & AGE IMPUTATION =====
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Enhanced title mapping with more categories
    title_mapping = {
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Mrs', 'Countess': 'Mrs', 'Capt': 'Officer',
        'Col': 'Officer', 'Don': 'Officer', 'Dr': 'Officer',
        'Major': 'Officer', 'Rev': 'Officer', 'Sir': 'Officer',
        'Jonkheer': 'Officer', 'Dona': 'Mrs'
    }
    df['Title'] = df['Title'].replace(title_mapping)
    
    # Fallback for remaining rare titles
    title_mask = ~df['Title'].isin(['Mr', 'Miss', 'Mrs', 'Master', 'Officer'])
    df.loc[title_mask, 'Title'] = df.loc[title_mask, 'Sex'].map({'male': 'Mr', 'female': 'Mrs'})
    
    # Enhanced age medians including Officer category
    title_age_medians = {
        'Mr': 32.32, 'Miss': 21.68, 'Mrs': 35.86, 
        'Master': 4.57, 'Officer': 49.0
    }
    
    for title, median_age in title_age_medians.items():
        age_mask = (df['Age'].isnull()) & (df['Title'] == title)
        df.loc[age_mask, 'Age'] = median_age
    
    # ===== CABIN & DECK FEATURES =====
    df['Deck'] = df['Cabin'].str[0]
    df['Deck'] = df['Deck'].fillna('Unknown')
    
    # Group rare decks
    deck_counts = df['Deck'].value_counts()
    rare_decks = deck_counts[deck_counts < 10].index
    df.loc[df['Deck'].isin(rare_decks), 'Deck'] = 'Rare'
    
    df['HasCabin'] = (~df['Cabin'].isnull()).astype(int)
    df['CabinCount'] = df['Cabin'].fillna('').str.count(' ') + 1
    df.loc[df['Cabin'].isnull(), 'CabinCount'] = 0
    
    # ===== TICKET FEATURES =====
    df['TicketPrefix'] = df['Ticket'].str.extract(r'([A-Za-z/\.]+)', expand=False)
    df['TicketPrefix'] = df['TicketPrefix'].fillna('None')
    
    # Group rare ticket prefixes
    prefix_counts = df['TicketPrefix'].value_counts()
    rare_prefixes = prefix_counts[prefix_counts < 10].index
    df.loc[df['TicketPrefix'].isin(rare_prefixes), 'TicketPrefix'] = 'Rare'
    
    df['TicketNumber'] = df['Ticket'].str.extract(r'(\d+)$', expand=False)
    df['TicketNumber'] = pd.to_numeric(df['TicketNumber'], errors='coerce').fillna(0)
    df['HasTicketNumber'] = (df['TicketNumber'] > 0).astype(int)
    
    # ===== FAMILY FEATURES =====
    df['Surname'] = df['Name'].str.extract(r'([A-Za-z]+),', expand=False)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Family size categories
    df['FamilySizeGroup'] = 'Medium'
    df.loc[df['FamilySize'] == 1, 'FamilySizeGroup'] = 'Alone'
    df.loc[df['FamilySize'] <= 4, 'FamilySizeGroup'] = 'Small'
    df.loc[df['FamilySize'] >= 5, 'FamilySizeGroup'] = 'Large'
    
    # ===== BASIC IMPUTATIONS =====
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # ===== INTERACTION FEATURES =====
    df['Age*Class'] = df['Age'] * df['Pclass']
    df['Age*Fare'] = df['Age'] * df['Fare']
    df['Fare*Class'] = df['Fare'] * df['Pclass']
    
    # Title-based features
    df['Title_Age_Ratio'] = df['Age'] / df.groupby('Title')['Age'].transform('mean')
    df['Title_Fare_Ratio'] = df['Fare'] / df.groupby('Title')['Fare'].transform('mean')
    
    # ===== BINNING FEATURES =====
    df['AgeBand'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, np.inf], labels=[0, 1, 2, 3, 4])
    df['AgeBand'] = df['AgeBand'].astype(int)
    
    df['FareBand'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
    df['FareBand'] = df['FareBand'].astype(int)
    
    # ===== TRANSFORMATIONS =====
    df['Fare_log'] = np.log1p(df['Fare'])
    df['Age_squared'] = df['Age'] ** 2
    df['Fare_sqrt'] = np.sqrt(df['Fare'])
    
    # ===== ONE-HOT ENCODING =====
    categorical_features = ['Sex', 'Pclass', 'Embarked', 'Title', 'Deck', 
                           'TicketPrefix', 'FamilySizeGroup']
    
    for feature in categorical_features:
        if feature in df.columns:
            dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=True, dtype=int)
            df = pd.concat([df, dummies], axis=1)
    
    # ===== FEATURE SELECTION & CLEANUP =====
    drop_columns = ['Sex', 'Pclass', 'Name', 'Ticket', 'Embarked', 'Cabin', 
                   'Title', 'Deck', 'TicketPrefix', 'FamilySizeGroup', 'Surname',
                   'SibSp', 'Parch', 'Fare']  # Keep Age for interactions
    
    existing_drops = [col for col in drop_columns if col in df.columns]
    df = df.drop(existing_drops, axis=1)
    
    # Remove 'Survived' if it exists (for test set)
    if 'Survived' in df.columns:
        df = df.drop('Survived', axis=1)
    
    # ===== FEATURE ALIGNMENT =====
    if feature_columns is not None:
        # Align test set features with training set
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0  # Add missing columns with default value
        
        # Reorder columns to match training set
        df = df[feature_columns]
    
    # ===== SCALING =====
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    if scaler is None:
        scaler = StandardScaler()
    
    if fit_scaler:
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        return df, scaler, list(df.columns)  # Return feature columns for test set
    else:
        df[numeric_columns] = scaler.transform(df[numeric_columns])
        return df, scaler


def create_family_survival_features(train_df, test_df=None):
    """
    Create family survival features based on training data.
    
    Args:
        train_df: Training DataFrame with 'Survived' column
        test_df: Optional test DataFrame
        
    Returns:
        Enhanced DataFrames with family survival features
    """
    # Extract surnames from training data
    train_df = train_df.copy()
    train_df['Surname'] = train_df['Name'].str.extract(r'([A-Za-z]+),', expand=False)
    
    # Calculate family survival rates
    family_survival = train_df.groupby('Surname').agg({
        'Survived': ['mean', 'count']
    }).round(3)
    
    family_survival.columns = ['FamilySurvivalRate', 'FamilyCount']
    family_survival = family_survival.reset_index()
    
    # Only use families with more than 1 member
    family_survival.loc[family_survival['FamilyCount'] == 1, 'FamilySurvivalRate'] = 0.5
    
    # Add to training data
    train_enhanced = train_df.merge(family_survival[['Surname', 'FamilySurvivalRate']], 
                                   on='Surname', how='left')
    train_enhanced['FamilySurvivalRate'] = train_enhanced['FamilySurvivalRate'].fillna(0.5)
    
    if test_df is not None:
        test_df = test_df.copy()
        test_df['Surname'] = test_df['Name'].str.extract(r'([A-Za-z]+),', expand=False)
        test_enhanced = test_df.merge(family_survival[['Surname', 'FamilySurvivalRate']], 
                                     on='Surname', how='left')
        test_enhanced['FamilySurvivalRate'] = test_enhanced['FamilySurvivalRate'].fillna(0.5)
        return train_enhanced, test_enhanced
    
    return train_enhanced