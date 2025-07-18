#!/usr/bin/env python3
"""
Rule-Based Enhanced Titanic Survival Predictor
Based on family survival patterns + advanced ML for remaining cases
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from src.data.enhanced_preprocessor import clean_transform_df, create_family_survival_features
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load training and test data"""
    try:
        train_df = pd.read_csv('titanic/train.csv')
        test_df = pd.read_csv('titanic/test.csv')
        return train_df, test_df
    except FileNotFoundError:
        print("Data files not found. Please ensure titanic/train.csv and titanic/test.csv exist.")
        return None, None

def extract_family_info(df):
    """Extract family information for rule-based predictions"""
    df = df.copy()
    
    # Extract surname and title
    df['Surname'] = df['Name'].str.extract(r'([A-Za-z]+),', expand=False)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Standardize titles
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
    
    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    return df

def apply_family_survival_rules(train_df, test_df):
    """Apply the specific family survival rules"""
    
    # Extract family info for both datasets
    train_enhanced = extract_family_info(train_df)
    test_enhanced = extract_family_info(test_df)
    
    # Calculate family survival rates from training data (excluding adult males)
    family_stats = []
    
    for surname in train_enhanced['Surname'].unique():
        family_members = train_enhanced[train_enhanced['Surname'] == surname]
        
        if len(family_members) > 1:  # Only families with multiple members
            # Exclude adult males (Mr title) for survival calculation
            non_adult_males = family_members[family_members['Title'] != 'Mr']
            
            if len(non_adult_males) > 0:
                survival_rate = non_adult_males['Survived'].mean()
                family_stats.append({
                    'Surname': surname,
                    'FamilySize': len(family_members),
                    'NonAdultMaleCount': len(non_adult_males),
                    'NonAdultMaleSurvivalRate': survival_rate,
                    'AllNonAdultMalesLive': survival_rate == 1.0,
                    'AllNonAdultMalesDie': survival_rate == 0.0
                })
    
    family_rules_df = pd.DataFrame(family_stats)
    
    # Initialize predictions with -1 (no rule applied)
    test_enhanced['RulePrediction'] = -1
    test_enhanced['RuleApplied'] = 'None'
    
    # Rule 1: Masters whose entire family (excluding adult males) all live
    rule1_families = family_rules_df[family_rules_df['AllNonAdultMalesLive']]['Surname'].tolist()
    
    rule1_mask = (
        (test_enhanced['Title'] == 'Master') & 
        (test_enhanced['Surname'].isin(rule1_families))
    )
    
    test_enhanced.loc[rule1_mask, 'RulePrediction'] = 1
    test_enhanced.loc[rule1_mask, 'RuleApplied'] = 'Rule1_MasterFamilyLives'
    
    # Rule 2: Females whose entire family (excluding adult males) all die
    rule2_families = family_rules_df[family_rules_df['AllNonAdultMalesDie']]['Surname'].tolist()
    
    rule2_mask = (
        (test_enhanced['Sex'] == 'female') & 
        (test_enhanced['Surname'].isin(rule2_families))
    )
    
    test_enhanced.loc[rule2_mask, 'RulePrediction'] = 0
    test_enhanced.loc[rule2_mask, 'RuleApplied'] = 'Rule2_FemaleFamilyDies'
    
    # Print rule applications
    rule1_count = rule1_mask.sum()
    rule2_count = rule2_mask.sum()
    
    print(f"Rule 1 (Master family lives): {rule1_count} predictions")
    if rule1_count > 0:
        rule1_passengers = test_enhanced[rule1_mask][['PassengerId', 'Name']].values
        for pid, name in rule1_passengers:
            print(f"  {pid}: {name}")
    
    print(f"\nRule 2 (Female family dies): {rule2_count} predictions")
    if rule2_count > 0:
        rule2_passengers = test_enhanced[rule2_mask][['PassengerId', 'Name']].values
        for pid, name in rule2_passengers:
            print(f"  {pid}: {name}")
    
    print(f"\nRemaining passengers for ML prediction: {(test_enhanced['RulePrediction'] == -1).sum()}")
    
    return test_enhanced, family_rules_df

def train_enhanced_model_for_remaining(train_df, test_df, test_with_rules):
    """Train model only on passengers not covered by rules"""
    
    # Create enhanced features for training
    train_enhanced = create_family_survival_features(train_df)
    
    # Process all data with enhanced preprocessing
    X_train, scaler, feature_columns = clean_transform_df(train_enhanced, fit_scaler=True)
    
    # For test set, only process passengers without rule predictions
    remaining_mask = test_with_rules['RulePrediction'] == -1
    test_remaining = test_df[remaining_mask].copy()
    
    if len(test_remaining) > 0:
        # Add family survival features to remaining test passengers
        test_remaining_enhanced = create_family_survival_features(train_df, test_remaining)[1]
        X_test_remaining, _ = clean_transform_df(test_remaining_enhanced, scaler=scaler, 
                                               fit_scaler=False, feature_columns=feature_columns)
    else:
        X_test_remaining = pd.DataFrame()
    
    y_train = train_enhanced['Survived']
    
    # Train ensemble of models
    models = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='logloss'
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_split=4,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trained_models = {}
    model_scores = {}
    
    print(f"\nTraining models on {len(X_train)} training samples...")
    print("-" * 50)
    
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        model_scores[name] = cv_scores
        
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        print(f"{name:15} CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Create ensemble predictions for remaining passengers
    if len(X_test_remaining) > 0:
        ensemble_pred = np.zeros(len(X_test_remaining))
        
        for name, model in trained_models.items():
            pred_proba = model.predict_proba(X_test_remaining)[:, 1]
            ensemble_pred += pred_proba
        
        ensemble_pred /= len(trained_models)
        ml_predictions = (ensemble_pred > 0.5).astype(int)
    else:
        ml_predictions = np.array([])
    
    return ml_predictions, remaining_mask, trained_models

def main():
    print("Rule-Based Enhanced Titanic Survival Predictor")
    print("=" * 60)
    
    # Load data
    train_df, test_df = load_data()
    if train_df is None:
        return
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Apply family survival rules
    print("\nApplying family survival rules...")
    test_with_rules, family_rules_df = apply_family_survival_rules(train_df, test_df)
    
    # Train model for remaining passengers
    ml_predictions, remaining_mask, trained_models = train_enhanced_model_for_remaining(
        train_df, test_df, test_with_rules
    )
    
    # Combine rule-based and ML predictions
    final_predictions = test_with_rules['RulePrediction'].copy()
    
    if len(ml_predictions) > 0:
        final_predictions.loc[remaining_mask] = ml_predictions
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': final_predictions.astype(int)
    })
    
    # Verify no missing predictions
    if (final_predictions == -1).any():
        print("WARNING: Some passengers have no prediction!")
        # Fallback: use gender-based prediction for any remaining
        missing_mask = final_predictions == -1
        gender_pred = (test_df.loc[missing_mask, 'Sex'] == 'female').astype(int)
        final_predictions.loc[missing_mask] = gender_pred
        submission['Survived'] = final_predictions.astype(int)
    
    submission.to_csv('rule_based_enhanced_submission.csv', index=False)
    
    # Summary statistics
    rule_predictions = (test_with_rules['RulePrediction'] != -1).sum()
    ml_predictions_count = len(ml_predictions)
    
    print(f"\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Rule-based predictions: {rule_predictions}")
    print(f"ML predictions: {ml_predictions_count}")
    print(f"Total predictions: {len(submission)}")
    print(f"Overall survival rate: {submission['Survived'].mean():.3f}")
    
    # Breakdown by prediction method
    rule1_count = (test_with_rules['RuleApplied'] == 'Rule1_MasterFamilyLives').sum()
    rule2_count = (test_with_rules['RuleApplied'] == 'Rule2_FemaleFamilyDies').sum()
    
    print(f"\nBreakdown:")
    print(f"  Rule 1 (Master lives): {rule1_count}")
    print(f"  Rule 2 (Female dies): {rule2_count}")
    print(f"  ML predictions: {ml_predictions_count}")
    
    if ml_predictions_count > 0:
        ml_survival_rate = ml_predictions.mean()
        print(f"  ML survival rate: {ml_survival_rate:.3f}")
    
    print(f"\nSubmission saved to 'rule_based_enhanced_submission.csv'")

if __name__ == "__main__":
    main()