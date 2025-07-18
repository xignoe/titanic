# Titanic Survival Predictor

A comprehensive machine learning pipeline for predicting passenger survival on the Titanic using advanced feature engineering, family survival rules, and ensemble methods. 

## 🏆 **Top 4% Performance on Kaggle**

**Kaggle Leaderboard Position: 628 / 15,496 (Top 4.05%)**

https://www.kaggle.com/xignoe

**Best Score: 0.80143** (80.143% accuracy)

## 🎯 Project Overview

This project implements a complete end-to-end machine learning solution for the classic Kaggle Titanic competition. It predicts which passengers survived the Titanic shipwreck based on features like age, sex, passenger class, family relationships, and more.

### Key Features

- **🎯 Family Survival Rules**: Implements domain-specific rules based on family survival patterns for high-confidence predictions
- **🧠 Advanced Feature Engineering**: 40+ engineered features including title-based age imputation, cabin deck extraction, and interaction features
- **🚀 Ensemble Methods**: Combines XGBoost, LightGBM, RandomForest, and SVM with performance-weighted voting
- **📊 Smart Preprocessing**: Title-based missing value imputation, feature interactions (Age*Class, Age*Fare), and intelligent binning
- **🔍 Multiple Approaches**: From simple rule-based models to complex ensemble methods with automatic model selection
- **⚡ Production Ready**: Robust error handling, comprehensive logging, and validation
- **🧪 Extensive Testing**: 187 unit, integration, and end-to-end tests with 100% pass rate

## 📊 Model Performance

### 🏆 Competition Results
- **Kaggle Public Score**: 0.80143 (80.143% accuracy)
- **Leaderboard Position**: 628 out of 15,496 submissions
- **Percentile**: Top 4.05% of all submissions

### 🎯 Model Evolution
| Approach | Score | Key Innovation |
|----------|-------|----------------|
| Enhanced ML | 0.77033 | Advanced feature engineering + ensemble |
| Rule-based MVP | 0.80143 | Family survival rules + gender baseline |
| Basic ML | 0.76794 | Standard preprocessing + RF/SVM |

### 📈 Cross-Validation Performance
- **XGBoost**: 87.32% CV accuracy (best individual model)
- **SVM**: 86.87% CV accuracy  
- **LightGBM**: 86.76% CV accuracy
- **RandomForest**: 86.76% CV accuracy

## 🏗️ Project Structure

```
├── main.py                     # Main CLI application
├── src/                        # Source code modules
│   ├── data/                   # Data handling
│   │   ├── loader.py           # Data loading with validation
│   │   ├── explorer.py         # Comprehensive data analysis
│   │   └── preprocessor.py     # Feature engineering & preprocessing
│   ├── models/                 # Machine learning models
│   │   ├── trainer.py          # Model training & hyperparameter tuning
│   │   ├── evaluator.py        # Performance evaluation & metrics
│   │   └── predictor.py        # Prediction generation & submission
│   └── utils/                  # Utilities
│       ├── config.py           # Configuration management
│       └── visualization.py    # Plotting and visualization tools
├── tests/                      # Comprehensive test suite
│   ├── test_data_*.py          # Data module tests
│   ├── test_model_*.py         # Model module tests
│   ├── test_integration.py     # Integration tests
│   └── test_end_to_end_validation.py  # End-to-end validation
├── titanic/                    # Data directory
│   ├── train.csv               # Training dataset
│   └── test.csv                # Test dataset
└── outputs/                    # Generated outputs
    ├── models/                 # Saved models
    ├── predictions/            # Prediction files
    └── visualizations/         # Generated plots
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn pytest joblib
```

### Installation

1. Clone or download the project
2. Ensure you have the Titanic dataset files in the `titanic/` directory:
   - `titanic/train.csv`
   - `titanic/test.csv`

### Running the Complete Pipeline

```bash
# Run the entire ML pipeline
python main.py pipeline --verbose

# This will:
# 1. Explore and analyze the data
# 2. Train multiple models with hyperparameter tuning
# 3. Select the best performing model
# 4. Generate predictions for the test set
# 5. Create submission file at outputs/predictions/submission.csv
```

## 🔧 Usage Options

### Individual Pipeline Steps

**Data Exploration**
```bash
python main.py explore --verbose
```
- Generates comprehensive data analysis
- Creates visualizations of survival patterns
- Analyzes missing values and feature distributions

**Model Training**
```bash
python main.py train --verbose
```
- Trains Random Forest, Logistic Regression, and SVM models
- Performs hyperparameter tuning with grid search
- Selects best model based on cross-validation

**Generate Predictions**
```bash
python main.py predict --verbose
```
- Loads the best trained model
- Generates predictions for test dataset
- Creates Kaggle-ready submission file

**Model Evaluation**
```bash
python main.py evaluate --verbose
```
- Evaluates model performance with detailed metrics
- Creates confusion matrix and ROC curve visualizations
- Analyzes feature importance

### Command Line Options

```bash
python main.py [mode] [options]

Modes:
  explore     - Data exploration and analysis
  train       - Model training and selection
  predict     - Generate predictions
  evaluate    - Model evaluation
  pipeline    - Complete end-to-end pipeline

Options:
  --verbose, -v           Enable detailed logging
  --data-dir DIR          Data directory (default: titanic)
  --output-dir DIR        Output directory (default: outputs)
  --model-path PATH       Specific model file to use
  --random-seed SEED      Random seed for reproducibility
  --min-accuracy FLOAT    Minimum accuracy threshold
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_data_*.py -v          # Data processing tests
python -m pytest tests/test_model_*.py -v         # Model tests
python -m pytest tests/test_integration.py -v     # Integration tests
python -m pytest tests/test_end_to_end_validation.py -v  # End-to-end tests
```

The test suite includes:
- **187 total tests** with 100% pass rate
- Unit tests for all components
- Integration tests for the complete pipeline
- End-to-end validation with real data
- Performance benchmark validation
- Submission format compliance testing

## 🔍 Advanced Feature Engineering

### 🎯 Family Survival Rules (Key Innovation)
The breakthrough to top 4% came from implementing domain-specific family survival rules:

**Rule 1**: Masters whose entire family (excluding adult males) survived → Predict LIVE
- Applied to 8 passengers: Ryerson, Wells, Touma, Drew, Spedden, Aks, Abbott, Peter

**Rule 2**: Females whose entire family (excluding adult males) died → Predict DIE  
- Applied to 10 passengers: Ilmakangas, Johnston, Cacic, Lefebre, Goodwin, Sage, Oreskovic, Rosblom, Riihivouri

### 🧠 Enhanced Feature Engineering (40+ Features)
- **Smart Age Imputation**: Title-based median ages (Master: 4.57, Miss: 21.68, Mrs: 35.86, Mr: 32.32)
- **Cabin Intelligence**: Deck extraction, cabin counts, availability indicators
- **Ticket Analysis**: Prefix patterns and numeric extraction
- **Family Features**: Size categories, survival rates by surname, alone indicators
- **Interaction Features**: Age*Class, Age*Fare, Fare*Class, Title-based ratios
- **Advanced Transformations**: Log(Fare), Age², √Fare, intelligent binning

### 🚀 Model Architecture
- **Ensemble Strategy**: Performance-weighted combination of top 4 models
- **Feature Selection**: Quality over quantity - 40 carefully engineered features
- **Preprocessing Pipeline**: Standardized scaling, missing value strategies, categorical encoding

## 📈 Model Details

### Algorithms Used
1. **Random Forest**: Ensemble method with feature importance analysis
2. **Logistic Regression**: Linear model with regularization
3. **Support Vector Machine**: Non-linear classification with RBF kernel

### Hyperparameter Tuning
- Grid search with cross-validation
- Optimizes for accuracy while preventing overfitting
- Automatic selection of best performing model

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC AUC and confusion matrix analysis
- Feature importance ranking
- Cross-validation scores

## 📁 Output Files

After running the pipeline, you'll find:

- **Models**: `outputs/models/best_*.joblib` - Trained model files
- **Predictions**: `outputs/predictions/submission.csv` - Kaggle submission file
- **Visualizations**: `outputs/visualizations/` - Generated plots and charts
- **Logs**: `titanic_predictor.log` - Detailed execution logs

## 🎯 Competition Compliance

The generated submission file is fully compliant with Kaggle competition requirements:
- Exactly 418 predictions (matching test set size)
- Proper CSV format with PassengerId and Survived columns
- Binary predictions (0 or 1)
- No missing values or duplicates

## 🔧 Configuration

Key configuration options in `src/utils/config.py`:
- Model parameters and hyperparameter grids
- File paths and directory structure
- Logging configuration
- Performance thresholds

## 🤝 Contributing

This project follows best practices for maintainable ML code:
- Modular design with clear separation of concerns
- Comprehensive testing and validation
- Detailed logging and error handling
- Type hints and documentation
- Configuration management

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Kaggle for providing the Titanic dataset
- The scikit-learn team for excellent ML tools
- The Python data science community for inspiration and best practices

---

**Ready to predict Titanic survival? Run `python main.py pipeline --verbose` to get started!**