#!/usr/bin/env python3
"""
Ultra Advanced Titanic Predictor - Minimal Rules + Best ML
Only apply the most confident family survival rules, then use our best ML approach
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

def apply_minimal_rules(train_df, test_df):
    """Apply ONLY the most confident family survival rules"""
    
    # Extract family info
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df['Surname'] = train_df['Name'].str.extract(r'([A-Za-z]+),', expand=False)
    train_df['Title'] = train_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    test_df['Surname'] = test_df['Name'].str.extract(r'([A-Za-z]+),', expand=False)
    test_df['Title'] = test_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Standardize titles
    title_mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}
    train_df['Title'] = train_df['Title'].replace(title_mapping)
    test_df['Title'] = test_df['Title'].replace(title_mapping)
    
    # Calculate family survival rates (excluding adult males)
    family_survival = {}
    
    for surname in train_df['Surname'].unique():
        family_members = train_df[train_df['Surname'] == surname]
        
        if len(family_members) > 1:  # Multi-member families only
            # Exclude adult males (Mr) for survival calculation
            non_adult_males = family_members[family_members['Title'] != 'Mr']
            
            if len(non_adult_males) > 0:
                survival_rate = non_adult_males['Survived'].mean()
                family_survival[surname] = {
                    'survival_rate': survival_rate,
                    'all_live': survival_rate == 1.0,
                    'all_die': survival_rate == 0.0,
                    'count': len(non_adult_males)
                }
    
    # Initialize rule predictions
    test_df['RulePrediction'] = -1
    
    # RULE 1: Masters whose family (non-adult males) all survived
    families_all_live = [name for name, data in family_survival.items() if data['all_live']]
    
    rule1_mask = (
        (test_df['Title'] == 'Master') & 
        (test_df['Surname'].isin(families_all_live))
    )
    test_df.loc[rule1_mask, 'RulePrediction'] = 1
    
    # RULE 2: Females whose family (non-adult males) all died
    families_all_die = [name for name, data in family_survival.items() if data['all_die']]
    
    rule2_mask = (
        (test_df['Sex'] == 'female') & 
        (test_df['Surname'].isin(families_all_die))
    )
    test_df.loc[rule2_mask, 'RulePrediction'] = 0
    
    # Print rule applications
    rule1_count = rule1_mask.sum()
    rule2_count = rule2_mask.sum()
    
    print(f"Rule 1 (Master family lives): {rule1_count} predictions")
    print(f"Rule 2 (Female family dies): {rule2_count} predictions")
    print(f"Total rule predictions: {rule1_count + rule2_count}")
    print(f"Remaining for ML: {(test_df['RulePrediction'] == -1).sum()}")
    
    return test_df

def main():
    print("Ultra Advanced Titanic Predictor - Minimal Rules + Best ML")
    print("=" * 65)
    
    # Load data
    train_df, test_df = load_data()
    if train_df is None:
        return
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Apply minimal family survival rules
    print("\nApplying minimal family survival rules...")
    test_with_rules = apply_minimal_rules(train_df, test_df)
    
    # Create enhanced features (our best preprocessing)
    print("\nCreating enhanced features...")
    train_enhanced, test_enhanced = create_family_survival_features(train_df, test_df)
    
    # Preprocess with our best approach
    X_train, scaler, feature_columns = clean_transform_df(train_enhanced, fit_scaler=True)
    X_test, _ = clean_transform_df(test_enhanced, scaler=scaler, fit_scaler=False, feature_columns=feature_columns)
    
    y_train = train_enhanced['Survived']
    
    print(f"Feature count: {X_train.shape[1]}")
    
    # Use our best models (from the 0.77033 submission)
    models = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5,
            random_state=42, eval_metric='logloss'
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5,
            random_state=42, verbose=-1
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=7, min_samples_split=6,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        )
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trained_models = {}
    model_scores = {}
    
    print("\nTraining models...")
    print("-" * 40)
    
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        model_scores[name] = cv_scores
        
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        print(f"{name:15} CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Create ensemble predictions
    ensemble_pred = np.zeros(len(X_test))
    
    # Weight by CV performance
    cv_means = {name: scores.mean() for name, scores in model_scores.items()}
    weights = np.array([cv_means[name] for name in models.keys()])
    weights = weights / weights.sum()
    
    for i, (name, model) in enumerate(trained_models.items()):
        pred_proba = model.predict_proba(X_test)[:, 1]
        ensemble_pred += weights[i] * pred_proba
    
    ml_predictions = (ensemble_pred > 0.5).astype(int)
    
    # Combine rule-based and ML predictions
    final_predictions = ml_predictions.copy()
    
    # Override with rule-based predictions where applicable
    rule_mask = test_with_rules['RulePrediction'] != -1
    final_predictions[rule_mask] = test_with_rules.loc[rule_mask, 'RulePrediction'].values
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': final_predictions
    })
    
    submission.to_csv('ultra_advanced_submission.csv', index=False)
    
    # Summary
    rule_count = rule_mask.sum()
    ml_count = len(final_predictions) - rule_count
    
    print(f"\n" + "=" * 65)
    print("PREDICTION SUMMARY")
    print("=" * 65)
    print(f"Rule-based predictions: {rule_count}")
    print(f"ML predictions: {ml_count}")
    print(f"Total predictions: {len(submission)}")
    print(f"Overall survival rate: {final_predictions.mean():.3f}")
    
    print(f"\nSubmission saved to 'ultra_advanced_submission.csv'")
    print("Goal: Beat 0.77033 with minimal rules + best ML")

if __name__ == "__main__":
    main()