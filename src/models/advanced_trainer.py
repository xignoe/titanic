"""
Advanced model training module with ensemble methods and modern algorithms.

This module provides sophisticated training strategies including stacking,
voting, boosting, and hyperparameter optimization.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from typing import Dict, Tuple, Any, Optional, List
import logging
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import advanced algorithms
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available. Install with: pip install catboost")


class AdvancedModelTrainer:
    """
    Advanced model trainer with ensemble methods and modern algorithms.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the AdvancedModelTrainer."""
        self.random_state = random_state
        self.models = {}
        self.model_scores = {}
        self.tuned_models = {}
        self.best_params = {}
        self.ensemble_models = {}
        
    def get_base_models(self) -> Dict[str, Any]:
        """Get base model configurations."""
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=2000,
                random_state=self.random_state,
                solver='liblinear'
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,  # Enable probability for stacking
                random_state=self.random_state
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance'
            ),
            'NaiveBayes': GaussianNB(),
            'DecisionTree': DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=self.random_state
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state
            )
        }
        
        # Add advanced algorithms if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss'
            )
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            )
        
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = cb.CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_seed=self.random_state,
                verbose=False
            )
        
        return models
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict[str, list]]:
        """Get hyperparameter grids for optimization."""
        grids = {
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'LogisticRegression': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'SVM': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'DecisionTree': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'AdaBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0, 2.0]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        # Add advanced algorithm grids
        if XGBOOST_AVAILABLE:
            grids['XGBoost'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        if LIGHTGBM_AVAILABLE:
            grids['LightGBM'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        if CATBOOST_AVAILABLE:
            grids['CatBoost'] = {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        
        return grids
    
    def train_base_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train all base models."""
        logger.info("Training base models...")
        
        models = self.get_base_models()
        trained_models = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                trained_models[name] = model
                logger.info(f"Successfully trained {name}")
            except Exception as e:
                logger.error(f"Failed to train {name}: {str(e)}")
                continue
        
        self.models = trained_models
        return trained_models
    
    def tune_hyperparameters_advanced(self, X_train: pd.DataFrame, y_train: pd.Series,
                                    cv_folds: int = 5, n_iter: int = 50) -> Dict[str, Any]:
        """Advanced hyperparameter tuning using RandomizedSearchCV."""
        logger.info("Starting advanced hyperparameter tuning...")
        
        if not self.models:
            raise ValueError("No models trained. Call train_base_models first.")
        
        grids = self.get_hyperparameter_grids()
        tuned_models = {}
        best_params = {}
        
        for name, model in self.models.items():
            if name not in grids:
                logger.warning(f"No parameter grid for {name}, using base model")
                tuned_models[name] = model
                continue
            
            logger.info(f"Tuning {name}...")
            try:
                # Use RandomizedSearchCV for efficiency
                search = RandomizedSearchCV(
                    estimator=type(model)(**model.get_params()),
                    param_distributions=grids[name],
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring='accuracy',
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=0
                )
                
                search.fit(X_train, y_train)
                
                tuned_models[name] = search.best_estimator_
                best_params[name] = search.best_params_
                
                logger.info(f"Best score for {name}: {search.best_score_:.4f}")
                logger.info(f"Best params for {name}: {search.best_params_}")
                
            except Exception as e:
                logger.error(f"Failed to tune {name}: {str(e)}")
                tuned_models[name] = model  # Fallback to base model
        
        self.tuned_models = tuned_models
        self.best_params = best_params
        return tuned_models
    
    def create_voting_ensemble(self, models: Dict[str, Any]) -> VotingClassifier:
        """Create a voting ensemble from the best models."""
        logger.info("Creating voting ensemble...")
        
        # Select top performing models for ensemble
        estimators = [(name, model) for name, model in models.items()]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability voting
            n_jobs=-1
        )
        
        return voting_clf
    
    def create_stacking_ensemble(self, models: Dict[str, Any], 
                               meta_classifier: Any = None) -> StackingClassifier:
        """Create a stacking ensemble."""
        logger.info("Creating stacking ensemble...")
        
        if meta_classifier is None:
            meta_classifier = LogisticRegression(random_state=self.random_state)
        
        # Select diverse models for stacking
        estimators = [(name, model) for name, model in models.items()]
        
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_classifier,
            cv=5,
            n_jobs=-1
        )
        
        return stacking_clf
    
    def train_ensemble_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train ensemble models."""
        logger.info("Training ensemble models...")
        
        if not self.tuned_models:
            raise ValueError("No tuned models available. Run hyperparameter tuning first.")
        
        ensemble_models = {}
        
        # Create and train voting ensemble
        try:
            voting_ensemble = self.create_voting_ensemble(self.tuned_models)
            voting_ensemble.fit(X_train, y_train)
            ensemble_models['VotingEnsemble'] = voting_ensemble
            logger.info("Successfully trained voting ensemble")
        except Exception as e:
            logger.error(f"Failed to train voting ensemble: {str(e)}")
        
        # Create and train stacking ensemble
        try:
            stacking_ensemble = self.create_stacking_ensemble(self.tuned_models)
            stacking_ensemble.fit(X_train, y_train)
            ensemble_models['StackingEnsemble'] = stacking_ensemble
            logger.info("Successfully trained stacking ensemble")
        except Exception as e:
            logger.error(f"Failed to train stacking ensemble: {str(e)}")
        
        # Create stacking with different meta-classifiers
        try:
            rf_meta = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            stacking_rf = self.create_stacking_ensemble(self.tuned_models, rf_meta)
            stacking_rf.fit(X_train, y_train)
            ensemble_models['StackingEnsemble_RF'] = stacking_rf
            logger.info("Successfully trained stacking ensemble with RF meta-classifier")
        except Exception as e:
            logger.error(f"Failed to train stacking ensemble with RF: {str(e)}")
        
        self.ensemble_models = ensemble_models
        return ensemble_models
    
    def evaluate_all_models(self, X_val: pd.DataFrame, y_val: pd.Series) -> pd.DataFrame:
        """Evaluate all models and return comprehensive results."""
        logger.info("Evaluating all models...")
        
        all_models = {}
        all_models.update(self.tuned_models)
        all_models.update(self.ensemble_models)
        
        results = []
        
        for name, model in all_models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_val)
                y_pred_proba = None
                
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                
                # Calculate AUC if probabilities available
                auc_score = None
                if y_pred_proba is not None:
                    try:
                        auc_score = roc_auc_score(y_val, y_pred_proba)
                    except:
                        pass
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_val, y_val, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                results.append({
                    'Model': name,
                    'Accuracy': accuracy,
                    'AUC': auc_score,
                    'CV_Mean': cv_mean,
                    'CV_Std': cv_std,
                    'Model_Type': 'Ensemble' if 'Ensemble' in name else 'Base'
                })
                
                logger.info(f"{name}: Accuracy={accuracy:.4f}, AUC={auc_score:.4f if auc_score else 'N/A'}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {str(e)}")
                continue
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        return results_df
    
    def select_best_model(self, X_val: pd.DataFrame, y_val: pd.Series, 
                         min_accuracy: float = 0.9) -> Tuple[str, Any, float, bool]:
        """Select the best performing model."""
        logger.info("Selecting best model...")
        
        results_df = self.evaluate_all_models(X_val, y_val)
        
        if len(results_df) == 0:
            raise ValueError("No models available for selection")
        
        # Get the best model
        best_row = results_df.iloc[0]
        best_name = best_row['Model']
        best_accuracy = best_row['Accuracy']
        
        # Get the model instance
        all_models = {}
        all_models.update(self.tuned_models)
        all_models.update(self.ensemble_models)
        
        best_model = all_models[best_name]
        meets_threshold = best_accuracy >= min_accuracy
        
        logger.info(f"Selected best model: {best_name}")
        logger.info(f"Accuracy: {best_accuracy:.4f}")
        logger.info(f"Meets {min_accuracy:.1%} threshold: {meets_threshold}")
        
        # Log top 5 models
        logger.info("Top 5 models:")
        for i, row in results_df.head().iterrows():
            logger.info(f"  {row['Model']}: {row['Accuracy']:.4f}")
        
        return best_name, best_model, best_accuracy, meets_threshold
    
    def save_model(self, model: Any, model_name: str, filepath: str = None) -> str:
        """Save a trained model."""
        if filepath is None:
            os.makedirs('outputs/models', exist_ok=True)
            filepath = f'outputs/models/{model_name}_advanced_model.joblib'
        
        try:
            joblib.dump(model, filepath)
            logger.info(f"Model {model_name} saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {str(e)}")
            raise
    
    def run_complete_training(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series,
                            min_accuracy: float = 0.9) -> Tuple[str, Any, float, bool]:
        """Run the complete advanced training pipeline."""
        logger.info("Starting complete advanced training pipeline...")
        
        # Step 1: Train base models
        self.train_base_models(X_train, y_train)
        
        # Step 2: Hyperparameter tuning
        self.tune_hyperparameters_advanced(X_train, y_train)
        
        # Step 3: Train ensemble models
        self.train_ensemble_models(X_train, y_train)
        
        # Step 4: Select best model
        best_name, best_model, best_accuracy, meets_threshold = self.select_best_model(
            X_val, y_val, min_accuracy
        )
        
        # Step 5: Save the best model
        self.save_model(best_model, f"best_{best_name}")
        
        logger.info("Complete advanced training pipeline finished")
        return best_name, best_model, best_accuracy, meets_threshold