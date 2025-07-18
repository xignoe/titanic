#!/usr/bin/env python3
"""
Simple Advanced Titanic Predictor
Take our BEST model (0.77033) and make the SMALLEST possible improvement
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
        print("Data files not found. Please ensure titanic/train.csv and titanic/test.csv exist.")
        return None, None

def train_ensemble_models(X_train, y_train):
    """Train the EXACT same models that got us 0.77033"""
    
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
    
    print("Training models (same as 0.77033 submission)...")
    print("-" * 50)
    
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        model_scores[name] = cv_scores
        
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        print(f"{name:20} CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return trained_models, model_scores

def create_ensemble_prediction(models, X_test, weights=None):
    """Create weighted ensemble predictions - EXACT same as 0.77033"""
    if weights is None:
        weights = [1.0] * len(models)
    
    predictions = np.zeros(len(X_test))
    
    for i, (name, model) in enumerate(models.items()):
        pred_proba = model.predict_proba(X_test)[:, 1]
        predictions += weights[i] * pred_proba
    
    predictions /= sum(weights)
    return (predictions > 0.5).astype(int)

def main():
    print("Simple Advanced Titanic Predictor")
    print("Replicating our BEST model (0.77033) with tiny improvements")
    print("=" * 60)
    
    # Load data
    train_df, test_df = load_data()
    if train_df is None:
        return
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Use EXACT same preprocessing as our best model
    print("\nUsing EXACT same preprocessing as 0.77033 model...")
    train_enhanced, test_enhanced = create_family_survival_features(train_df, test_df)
    
    X_train, scaler, feature_columns = clean_transform_df(train_enhanced, fit_scaler=True)
    X_test, _ = clean_transform_df(test_enhanced, scaler=scaler, fit_scaler=False, feature_columns=feature_columns)
    
    y_train = train_enhanced['Survived']
    
    print(f"Feature count: {X_train.shape[1]} (same as before)")
    
    # Train EXACT same models
    trained_models, model_scores = train_ensemble_models(X_train, y_train)
    
    # Use EXACT same ensemble strategy
    cv_means = {name: scores.mean() for name, scores in model_scores.items()}
    best_models = sorted(cv_means.items(), key=lambda x: x[1], reverse=True)[:4]
    
    print(f"\nTop 4 models for ensemble (same selection as 0.77033):")
    ensemble_models = {}
    weights = []
    
    for name, score in best_models:
        ensemble_models[name] = trained_models[name]
        weights.append(score)
        print(f"{name}: {score:.4f}")
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # ONLY CHANGE: Slightly adjust the prediction threshold
    # Instead of 0.5, use 0.49 (slightly more liberal)
    predictions = np.zeros(len(X_test))
    
    for i, (name, model) in enumerate(ensemble_models.items()):
        pred_proba = model.predict_proba(X_test)[:, 1]
        predictions += weights[i] * pred_proba
    
    predictions /= sum(weights)
    
    # TINY adjustment: 0.49 instead of 0.5
    ensemble_pred = (predictions > 0.49).astype(int)
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': ensemble_pred
    })
    
    submission.to_csv('simple_advanced_submission.csv', index=False)
    
    print(f"\nSubmission saved to 'simple_advanced_submission.csv'")
    print(f"Predicted survival rate: {ensemble_pred.mean():.3f}")
    print(f"ONLY change: threshold 0.49 instead of 0.5")
    print(f"Goal: Beat 0.77033 with minimal change")

if __name__ == "__main__":
    main()