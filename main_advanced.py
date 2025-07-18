#!/usr/bin/env python3
"""
Advanced Titanic Predictor - Rules + Best ML
Combine 17 family rules with our best ML approach (0.77033)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
        print("Data files not found.")
        return None, None

def apply_17_rules(train_df, test_df):
    """Apply the 17 family survival rules we can detect"""
    
    # Extract family info from training data
    train_df = train_df.copy()
    train_df['Surname'] = train_df['Name'].str.extract(r'([A-Za-z]+),', expand=False)
    train_df['Title'] = train_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Calculate family survival (excluding adult males = Mr)
    family_survival = {}
    for surname in train_df['Surname'].unique():
        family = train_df[train_df['Surname'] == surname]
        non_adult_males = family[family['Title'] != 'Mr']
        if len(non_adult_males) > 0:
            survival_rate = non_adult_males['Survived'].mean()
            family_survival[surname] = {
                'all_live': survival_rate == 1.0,
                'all_die': survival_rate == 0.0
            }
    
    # Extract info from test data
    test_df = test_df.copy()
    test_df['Surname'] = test_df['Name'].str.extract(r'([A-Za-z]+),', expand=False)
    test_df['Title'] = test_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Initialize predictions
    test_df['RulePrediction'] = -1
    
    # RULE 1: Masters whose family all live → LIVE
    families_all_live = [name for name, data in family_survival.items() if data['all_live']]
    rule1_mask = (test_df['Title'] == 'Master') & (test_df['Surname'].isin(families_all_live))
    test_df.loc[rule1_mask, 'RulePrediction'] = 1
    
    # RULE 2: Females whose family all die → DIE  
    families_all_die = [name for name, data in family_survival.items() if data['all_die']]
    rule2_mask = (test_df['Sex'] == 'female') & (test_df['Surname'].isin(families_all_die))
    test_df.loc[rule2_mask, 'RulePrediction'] = 0
    
    rule_count = (test_df['RulePrediction'] != -1).sum()
    print(f"Applied 17 family survival rules to {rule_count} passengers")
    
    return test_df

def main():
    print("Advanced Titanic Predictor - Rules + Best ML")
    print("17 family rules + enhanced ML (goal: beat 0.78708)")
    print("=" * 60)
    
    # Load data
    train_df, test_df = load_data()
    if train_df is None:
        return
    
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    
    # Apply 17 family survival rules
    print(f"\nApplying 17 family survival rules...")
    test_with_rules = apply_17_rules(train_df, test_df)
    
    # Use our BEST preprocessing (0.77033 approach)
    print(f"\nUsing best preprocessing (enhanced features)...")
    train_enhanced, test_enhanced = create_family_survival_features(train_df, test_df)
    
    X_train, scaler, feature_columns = clean_transform_df(train_enhanced, fit_scaler=True)
    X_test, _ = clean_transform_df(test_enhanced, scaler=scaler, fit_scaler=False, feature_columns=feature_columns)
    
    y_train = train_enhanced['Survived']
    
    print(f"Feature count: {X_train.shape[1]}")
    
    # Use our BEST ensemble (from 0.77033)
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=7, min_samples_split=6,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5,
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5,
            random_state=42, eval_metric='logloss'
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5,
            random_state=42, verbose=-1
        ),
        'LogisticRegression': LogisticRegression(
            C=1.0, random_state=42, max_iter=1000
        ),
        'SVM': SVC(
            C=1.0, kernel='rbf', probability=True, random_state=42
        )
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trained_models = {}
    model_scores = {}
    
    print("Training best ensemble...")
    print("-" * 40)
    
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        model_scores[name] = cv_scores
        
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        print(f"{name:20} CV: {cv_scores.mean():.4f}")
    
    # Create ensemble predictions for ALL passengers
    cv_means = {name: scores.mean() for name, scores in model_scores.items()}
    best_models = sorted(cv_means.items(), key=lambda x: x[1], reverse=True)[:4]
    
    ensemble_models = {}
    weights = []
    
    for name, score in best_models:
        ensemble_models[name] = trained_models[name]
        weights.append(score)
    
    weights = np.array(weights) / np.sum(weights)
    
    # Get ML predictions for all passengers
    ml_predictions = np.zeros(len(X_test))
    
    for i, (name, model) in enumerate(ensemble_models.items()):
        pred_proba = model.predict_proba(X_test)[:, 1]
        ml_predictions += weights[i] * pred_proba
    
    ml_predictions = (ml_predictions > 0.5).astype(int)
    
    # Combine: Use rules where available, ML elsewhere
    final_predictions = ml_predictions.copy()
    
    rule_mask = test_with_rules['RulePrediction'] != -1
    final_predictions[rule_mask] = test_with_rules.loc[rule_mask, 'RulePrediction'].values
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': final_predictions
    })
    
    submission.to_csv('advanced_submission.csv', index=False)
    
    # Summary
    rule_count = rule_mask.sum()
    ml_count = len(final_predictions) - rule_count
    
    print(f"\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Rule-based predictions: {rule_count}")
    print(f"ML predictions: {ml_count}")
    print(f"Overall survival rate: {final_predictions.mean():.3f}")
    print(f"\nSubmission saved to 'advanced_submission.csv'")
    print("Goal: Beat 0.78708 (current best)")

if __name__ == "__main__":
    main()