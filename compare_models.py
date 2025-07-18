#!/usr/bin/env python3
"""
Compare different Titanic model implementations
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.data.preprocessor import DataPreprocessor
from src.data.enhanced_preprocessor import clean_transform_df as enhanced_preprocess, create_family_survival_features
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load training data"""
    try:
        train_df = pd.read_csv('titanic/train.csv')
        return train_df
    except FileNotFoundError:
        print("Data files not found. Please ensure titanic/train.csv exists.")
        return None

def evaluate_preprocessing(train_df, preprocess_approach, use_family_features=False):
    """Evaluate a preprocessing approach"""
    
    if use_family_features:
        train_enhanced = create_family_survival_features(train_df)
        data_to_process = train_enhanced
    else:
        data_to_process = train_df
    
    if preprocess_approach == "basic":
        # Use basic preprocessor class
        preprocessor = DataPreprocessor()
        preprocessor.fit(data_to_process)
        X_train, _ = preprocessor.preprocess_data(data_to_process)
    elif preprocess_approach == "enhanced":
        # Use enhanced preprocessor function
        X_train, _, _ = enhanced_preprocess(data_to_process, fit_scaler=True)
    
    y_train = train_df['Survived']
    
    # Use RandomForest for consistent comparison
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    return cv_scores, X_train.shape[1]

def main():
    print("Titanic Model Comparison")
    print("=" * 50)
    
    # Load data
    train_df = load_data()
    if train_df is None:
        return
    
    print(f"Training data shape: {train_df.shape}")
    print()
    
    # Test different preprocessing approaches
    approaches = [
        ("Basic Preprocessing", "basic", False),
        ("Enhanced Preprocessing", "enhanced", False),
        ("Enhanced + Family Features", "enhanced", True),
    ]
    
    results = []
    
    for name, preprocess_approach, use_family in approaches:
        print(f"Testing {name}...")
        try:
            cv_scores, n_features = evaluate_preprocessing(train_df, preprocess_approach, use_family)
            
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            results.append({
                'Approach': name,
                'CV Score': f"{mean_score:.4f} (+/- {std_score * 2:.4f})",
                'Features': n_features,
                'Mean': mean_score
            })
            
            print(f"  CV Score: {mean_score:.4f} (+/- {std_score * 2:.4f})")
            print(f"  Features: {n_features}")
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            print()
    
    # Summary table
    print("Summary Comparison:")
    print("-" * 60)
    print(f"{'Approach':<25} {'CV Score':<20} {'Features':<10}")
    print("-" * 60)
    
    # Sort by performance
    results.sort(key=lambda x: x['Mean'], reverse=True)
    
    for result in results:
        print(f"{result['Approach']:<25} {result['CV Score']:<20} {result['Features']:<10}")
    
    print("-" * 60)
    
    if len(results) > 1:
        best = results[0]
        baseline = results[-1]
        improvement = best['Mean'] - baseline['Mean']
        print(f"\nBest approach improves accuracy by {improvement:.4f} ({improvement*100:.2f}%)")

if __name__ == "__main__":
    main()