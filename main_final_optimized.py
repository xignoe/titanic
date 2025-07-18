#!/usr/bin/env python3
"""
Final Optimized Titanic Survival Predictor
Rule-based + Advanced ML + Gender fallback to beat 82.3%
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
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
    
    # Additional Rule 3: High-confidence predictions based on training patterns
    # Masters with family size > 1 but not covered by Rule 1 - tend to survive
    rule3_mask = (
        (test_enhanced['Title'] == 'Master') & 
        (test_enhanced['FamilySize'] > 1) &
        (test_enhanced['RulePrediction'] == -1)
    )
    test_enhanced.loc[rule3_mask, 'RulePrediction'] = 1
    test_enhanced.loc[rule3_mask, 'RuleApplied'] = 'Rule3_MasterWithFamily'
    
    # Rule 4: First class females with family - very high survival rate
    rule4_mask = (
        (test_enhanced['Sex'] == 'female') & 
        (test_enhanced['Pclass'] == 1) &
        (test_enhanced['FamilySize'] > 1) &
        (test_enhanced['RulePrediction'] == -1)
    )
    test_enhanced.loc[rule4_mask, 'RulePrediction'] = 1
    test_enhanced.loc[rule4_mask, 'RuleApplied'] = 'Rule4_FirstClassFemaleWithFamily'
    
    # Print rule applications
    for rule_name in ['Rule1_MasterFamilyLives', 'Rule2_FemaleFamilyDies', 'Rule3_MasterWithFamily', 'Rule4_FirstClassFemaleWithFamily']:
        rule_count = (test_enhanced['RuleApplied'] == rule_name).sum()
        print(f"{rule_name}: {rule_count} predictions")
        
        if rule_count > 0 and rule_count <= 10:  # Only show details for small numbers
            rule_passengers = test_enhanced[test_enhanced['RuleApplied'] == rule_name][['PassengerId', 'Name']].values
            for pid, name in rule_passengers:
                print(f"  {pid}: {name}")
    
    total_rules = (test_enhanced['RulePrediction'] != -1).sum()
    print(f"\nTotal rule-based predictions: {total_rules}")
    print(f"Remaining passengers for ML prediction: {(test_enhanced['RulePrediction'] == -1).sum()}")
    
    return test_enhanced, family_rules_df

def train_advanced_ensemble(train_df, test_df, test_with_rules):
    """Train advanced ensemble for remaining passengers"""
    
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
    
    # Advanced ensemble with diverse models
    models = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=7,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            random_state=42, eval_metric='logloss'
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=7,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            random_state=42, verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=300, learning_rate=0.03, depth=7,
            random_seed=42, verbose=False
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=400, max_depth=10, min_samples_split=3,
            min_samples_leaf=1, random_state=42, n_jobs=-1
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=400, max_depth=10, min_samples_split=3,
            min_samples_leaf=1, random_state=42, n_jobs=-1
        ),
        'LogisticRegression': LogisticRegression(
            C=0.5, random_state=42, max_iter=2000
        )
    }
    
    cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    trained_models = {}
    model_scores = {}
    
    print(f"\nTraining advanced ensemble on {len(X_train)} training samples...")
    print("-" * 60)
    
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        model_scores[name] = cv_scores
        
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        print(f"{name:20} CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Create weighted ensemble predictions
    if len(X_test_remaining) > 0:
        # Calculate weights based on CV performance
        cv_means = {name: scores.mean() for name, scores in model_scores.items()}
        weights = np.array([cv_means[name] for name in models.keys()])
        weights = weights / weights.sum()
        
        ensemble_pred = np.zeros(len(X_test_remaining))
        
        for i, (name, model) in enumerate(trained_models.items()):
            pred_proba = model.predict_proba(X_test_remaining)[:, 1]
            ensemble_pred += weights[i] * pred_proba
        
        # Use dynamic threshold based on gender for remaining passengers
        remaining_test_data = test_df[remaining_mask]
        ml_predictions = np.zeros(len(X_test_remaining), dtype=int)
        
        # Different thresholds for males vs females
        male_mask = remaining_test_data['Sex'] == 'male'
        female_mask = remaining_test_data['Sex'] == 'female'
        
        # More conservative threshold for males (harder to survive)
        ml_predictions[male_mask] = (ensemble_pred[male_mask] > 0.6).astype(int)
        # More liberal threshold for females (easier to survive)
        ml_predictions[female_mask] = (ensemble_pred[female_mask] > 0.4).astype(int)
        
    else:
        ml_predictions = np.array([])
    
    return ml_predictions, remaining_mask, trained_models, model_scores

def main():
    print("Final Optimized Titanic Survival Predictor")
    print("=" * 70)
    
    # Load data
    train_df, test_df = load_data()
    if train_df is None:
        return
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Apply enhanced family survival rules
    print("\nApplying enhanced family survival rules...")
    test_with_rules, family_rules_df = apply_family_survival_rules(train_df, test_df)
    
    # Train advanced ensemble for remaining passengers
    ml_predictions, remaining_mask, trained_models, model_scores = train_advanced_ensemble(
        train_df, test_df, test_with_rules
    )
    
    # Combine rule-based and ML predictions
    final_predictions = test_with_rules['RulePrediction'].copy()
    
    if len(ml_predictions) > 0:
        final_predictions.loc[remaining_mask] = ml_predictions
    
    # Final fallback: gender-based prediction for any remaining
    if (final_predictions == -1).any():
        print("Applying gender-based fallback for remaining passengers...")
        missing_mask = final_predictions == -1
        gender_pred = (test_df.loc[missing_mask, 'Sex'] == 'female').astype(int)
        final_predictions.loc[missing_mask] = gender_pred
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': final_predictions.astype(int)
    })
    
    submission.to_csv('titanic_final_optimized.csv', index=False)
    
    # Detailed summary statistics
    rule_predictions = (test_with_rules['RulePrediction'] != -1).sum()
    ml_predictions_count = len(ml_predictions)
    
    print(f"\n" + "=" * 70)
    print("FINAL PREDICTION SUMMARY")
    print("=" * 70)
    
    # Breakdown by prediction method
    for rule_name in ['Rule1_MasterFamilyLives', 'Rule2_FemaleFamilyDies', 'Rule3_MasterWithFamily', 'Rule4_FirstClassFemaleWithFamily']:
        count = (test_with_rules['RuleApplied'] == rule_name).sum()
        survival_rate = test_with_rules[test_with_rules['RuleApplied'] == rule_name]['RulePrediction'].mean() if count > 0 else 0
        print(f"{rule_name:30} {count:3d} predictions (survival: {survival_rate:.1f})")
    
    print(f"{'ML predictions':30} {ml_predictions_count:3d} predictions (survival: {ml_predictions.mean():.3f})")
    print(f"{'Total predictions':30} {len(submission):3d}")
    print(f"{'Overall survival rate':30} {submission['Survived'].mean():.3f}")
    
    # Model performance summary
    print(f"\nTop model CV scores:")
    cv_means = {name: scores.mean() for name, scores in model_scores.items()}
    for name, score in sorted(cv_means.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:20} {score:.4f}")
    
    print(f"\nSubmission saved to 'titanic_final_optimized.csv'")
    print("Target: Beat 82.3% (0.823) on Kaggle leaderboard")

if __name__ == "__main__":
    main()