#!/usr/bin/env python3
"""
MVP Rules-Based Titanic Predictor - FIXED VERSION
Correctly implement all 18 rules + gender baseline to beat 82.3%
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load training and test data"""
    try:
        train_df = pd.read_csv('titanic/train.csv')
        test_df = pd.read_csv('titanic/test.csv')
        return train_df, test_df
    except FileNotFoundError:
        print("Data files not found.")
        return None, None

def apply_exact_rules_fixed(train_df, test_df):
    """Apply the EXACT 18 rules specified - FIXED VERSION"""
    
    # Extract family info from training data
    train_df = train_df.copy()
    train_df['Surname'] = train_df['Name'].str.extract(r'([A-Za-z]+),', expand=False)
    train_df['Title'] = train_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Calculate family survival (excluding adult males = Mr)
    family_survival = {}
    for surname in train_df['Surname'].unique():
        family = train_df[train_df['Surname'] == surname]
        # FIXED: Include single-member families too
        non_adult_males = family[family['Title'] != 'Mr']
        if len(non_adult_males) > 0:
            survival_rate = non_adult_males['Survived'].mean()
            family_survival[surname] = {
                'all_live': survival_rate == 1.0,
                'all_die': survival_rate == 0.0,
                'count': len(non_adult_males)
            }
    
    # Extract info from test data
    test_df = test_df.copy()
    test_df['Surname'] = test_df['Name'].str.extract(r'([A-Za-z]+),', expand=False)
    test_df['Title'] = test_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Initialize predictions
    test_df['Prediction'] = -1  # -1 means no rule applied
    
    # RULE 1: Masters whose family (non-adult males) all live → LIVE
    families_all_live = [name for name, data in family_survival.items() if data['all_live']]
    rule1_mask = (test_df['Title'] == 'Master') & (test_df['Surname'].isin(families_all_live))
    test_df.loc[rule1_mask, 'Prediction'] = 1
    
    # RULE 2: Females whose family (non-adult males) all die → DIE  
    families_all_die = [name for name, data in family_survival.items() if data['all_die']]
    rule2_mask = (test_df['Sex'] == 'female') & (test_df['Surname'].isin(families_all_die))
    test_df.loc[rule2_mask, 'Prediction'] = 0
    
    # Print the exact passengers affected
    print("RULE 1 - Masters whose family all live (predict LIVE):")
    rule1_passengers = test_df[rule1_mask][['PassengerId', 'Name']]
    for _, row in rule1_passengers.iterrows():
        print(f"  {row['PassengerId']}: {row['Name']}")
    
    print(f"\nRULE 2 - Females whose family all die (predict DIE):")
    rule2_passengers = test_df[rule2_mask][['PassengerId', 'Name']]
    for _, row in rule2_passengers.iterrows():
        print(f"  {row['PassengerId']}: {row['Name']}")
    
    rule_count = (test_df['Prediction'] != -1).sum()
    remaining_count = (test_df['Prediction'] == -1).sum()
    
    print(f"\nTotal rule predictions: {rule_count}")
    print(f"Remaining for gender rule: {remaining_count}")
    
    return test_df

def apply_gender_baseline(test_df):
    """Apply simple gender rule: males die, females live"""
    
    # For passengers without rule predictions, use gender
    no_rule_mask = test_df['Prediction'] == -1
    
    # Males die (0), Females live (1)
    test_df.loc[no_rule_mask & (test_df['Sex'] == 'male'), 'Prediction'] = 0
    test_df.loc[no_rule_mask & (test_df['Sex'] == 'female'), 'Prediction'] = 1
    
    return test_df

def improve_with_best_features(train_df, test_df):
    """Use our best features to improve remaining predictions"""
    
    # Only work on passengers without rule predictions
    no_rule_mask = test_df['Prediction'] == -1
    remaining_test = test_df[no_rule_mask].copy()
    
    if len(remaining_test) == 0:
        return test_df
    
    # Use key features from our best model (0.77033)
    # Extract titles for better age imputation
    train_df['Title'] = train_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    remaining_test['Title'] = remaining_test['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Title-based age imputation (from our best model)
    title_age_medians = {
        'Mr': 32.32, 'Miss': 21.68, 'Mrs': 35.86, 
        'Master': 4.57, 'Dr': 49.0, 'Rev': 49.0
    }
    
    for title, median_age in title_age_medians.items():
        train_mask = (train_df['Age'].isnull()) & (train_df['Title'] == title)
        train_df.loc[train_mask, 'Age'] = median_age
        
        test_mask = (remaining_test['Age'].isnull()) & (remaining_test['Title'] == title)
        remaining_test.loc[test_mask, 'Age'] = median_age
    
    # Fill remaining missing values
    train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
    train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
    train_df['Embarked'] = train_df['Embarked'].fillna('S')
    
    remaining_test['Age'] = remaining_test['Age'].fillna(remaining_test['Age'].median())
    remaining_test['Fare'] = remaining_test['Fare'].fillna(remaining_test['Fare'].median())
    remaining_test['Embarked'] = remaining_test['Embarked'].fillna('S')
    
    # Key features from our best model
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)
    train_df['Sex_male'] = (train_df['Sex'] == 'male').astype(int)
    train_df['Age*Class'] = train_df['Age'] * train_df['Pclass']
    
    remaining_test['FamilySize'] = remaining_test['SibSp'] + remaining_test['Parch'] + 1
    remaining_test['IsAlone'] = (remaining_test['FamilySize'] == 1).astype(int)
    remaining_test['Sex_male'] = (remaining_test['Sex'] == 'male').astype(int)
    remaining_test['Age*Class'] = remaining_test['Age'] * remaining_test['Pclass']
    
    # Select best features
    features = ['Pclass', 'Sex_male', 'Age', 'FamilySize', 'IsAlone', 'Fare', 'Age*Class']
    
    X_train = train_df[features]
    y_train = train_df['Survived']
    X_remaining = remaining_test[features]
    
    # Use XGBoost (our best individual model)
    model = xgb.XGBClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=5,
        random_state=42, eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # Predict for remaining passengers
    ml_predictions = model.predict(X_remaining)
    
    # Update test_df with ML predictions
    test_df.loc[no_rule_mask, 'Prediction'] = ml_predictions
    
    print(f"Applied XGBoost to {len(remaining_test)} remaining passengers")
    
    return test_df

def main():
    print("MVP Rules-Based Titanic Predictor - FIXED VERSION")
    print("Goal: Find all 18 rule passengers + beat 82.3%")
    print("=" * 60)
    
    # Load data
    train_df, test_df = load_data()
    if train_df is None:
        return
    
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    
    # Step 1: Apply the exact 18 rules (FIXED)
    print(f"\nStep 1: Applying FIXED family survival rules...")
    test_with_rules = apply_exact_rules_fixed(train_df, test_df)
    
    # Step 2: Apply gender baseline for remaining
    print(f"\nStep 2: Applying gender rule for remaining passengers...")
    test_with_gender = apply_gender_baseline(test_with_rules)
    
    # Create baseline submission (should score 82.3%)
    baseline_submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': test_with_gender['Prediction']
    })
    baseline_submission.to_csv('baseline_823_submission.csv', index=False)
    
    # Step 3: Use best ML features for improvement
    print(f"\nStep 3: Using best ML features for remaining passengers...")
    test_improved = improve_with_best_features(train_df, test_with_rules.copy())
    
    # Create improved submission
    improved_submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': test_improved['Prediction']
    })
    improved_submission.to_csv('mvp_improved_submission.csv', index=False)
    
    # Summary
    rule_count = (test_with_rules['Prediction'] != -1).sum()
    baseline_survival = baseline_submission['Survived'].mean()
    improved_survival = improved_submission['Survived'].mean()
    
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Rule predictions: {rule_count}")
    print(f"Gender baseline survival rate: {baseline_survival:.3f}")
    print(f"Improved model survival rate: {improved_survival:.3f}")
    print(f"\nFiles created:")
    print(f"  baseline_823_submission.csv (should score ~82.3%)")
    print(f"  mvp_improved_submission.csv (goal: beat 82.3%)")

if __name__ == "__main__":
    main()