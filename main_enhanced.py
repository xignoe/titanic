#!/usr/bin/env python3
"""
Enhanced Titanic Survival Predictor with Advanced Feature Engineering
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    """Train multiple models and return them with their CV scores"""
    
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
    
    print("Training and evaluating models...")
    print("-" * 50)
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        model_scores[name] = cv_scores
        
        # Train on full dataset
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        print(f"{name:20} CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return trained_models, model_scores

def create_ensemble_prediction(models, X_test, weights=None):
    """Create weighted ensemble predictions"""
    if weights is None:
        weights = [1.0] * len(models)
    
    predictions = np.zeros(len(X_test))
    
    for i, (name, model) in enumerate(models.items()):
        pred_proba = model.predict_proba(X_test)[:, 1]
        predictions += weights[i] * pred_proba
    
    predictions /= sum(weights)
    return (predictions > 0.5).astype(int)

def main():
    print("Enhanced Titanic Survival Predictor")
    print("=" * 50)
    
    # Load data
    train_df, test_df = load_data()
    if train_df is None:
        return
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Create family survival features
    print("\nCreating family survival features...")
    train_enhanced, test_enhanced = create_family_survival_features(train_df, test_df)
    
    # Preprocess data
    print("Preprocessing data with enhanced features...")
    X_train, scaler, feature_columns = clean_transform_df(train_enhanced, fit_scaler=True)
    X_test, _ = clean_transform_df(test_enhanced, scaler=scaler, fit_scaler=False, feature_columns=feature_columns)
    
    y_train = train_enhanced['Survived']
    
    print(f"Final feature count: {X_train.shape[1]}")
    print(f"Feature names: {list(X_train.columns)}")
    
    # Train models
    trained_models, model_scores = train_ensemble_models(X_train, y_train)
    
    # Calculate ensemble weights based on CV performance
    cv_means = {name: scores.mean() for name, scores in model_scores.items()}
    best_models = sorted(cv_means.items(), key=lambda x: x[1], reverse=True)[:4]
    
    print(f"\nTop 4 models for ensemble:")
    ensemble_models = {}
    weights = []
    
    for name, score in best_models:
        ensemble_models[name] = trained_models[name]
        weights.append(score)
        print(f"{name}: {score:.4f}")
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Make predictions
    print(f"\nCreating ensemble predictions with weights: {dict(zip(ensemble_models.keys(), weights))}")
    ensemble_pred = create_ensemble_prediction(ensemble_models, X_test, weights)
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': ensemble_pred
    })
    
    submission.to_csv('enhanced_titanic_submission.csv', index=False)
    print(f"\nSubmission saved to 'enhanced_titanic_submission.csv'")
    print(f"Predicted survival rate: {ensemble_pred.mean():.3f}")
    
    # Feature importance analysis
    print(f"\nTop 15 Feature Importances (Random Forest):")
    rf_model = trained_models['RandomForest']
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(15).iterrows():
        print(f"{row['feature']:25} {row['importance']:.4f}")

if __name__ == "__main__":
    main()