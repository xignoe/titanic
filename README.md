# Titanic Survival Predictor

A comprehensive machine learning pipeline for predicting passenger survival on the Titanic using feature engineering and multiple model evaluation. Result submitted on kaggle:

https://www.kaggle.com/xignoe

Score: 0.76794

## 🎯 Project Overview

This project implements a complete end-to-end machine learning solution for the classic Kaggle Titanic competition. It predicts which passengers survived the Titanic shipwreck based on features like age, sex, passenger class, family relationships, and more.

### Key Features

- **Complete ML Pipeline**: End-to-end workflow from data exploration to prediction submission
- **Advanced Feature Engineering**: Creates meaningful features from raw data (titles, family size, deck extraction)
- **Multiple Model Training**: Trains and compares Random Forest, Logistic Regression, and SVM models
- **Hyperparameter Tuning**: Automated grid search for optimal model parameters
- **Comprehensive Evaluation**: Detailed performance metrics, confusion matrices, and ROC analysis
- **Production Ready**: Robust error handling, logging, and validation
- **Extensive Testing**: 187 unit, integration, and end-to-end tests with 100% pass rate

## 📊 Model Performance

The pipeline achieves strong performance across multiple metrics:
- **Accuracy**: ~84%
- **Precision**: ~80%
- **Recall**: ~75%
- **F1 Score**: ~77%
- **ROC AUC**: ~85%

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

## 🔍 Feature Engineering

The pipeline creates several engineered features:

- **Title Extraction**: Extracts titles (Mr, Mrs, Miss, etc.) from passenger names
- **Family Size**: Combines SibSp and Parch to create family size features
- **Is Alone**: Binary feature indicating if passenger traveled alone
- **Deck Information**: Extracts deck information from cabin numbers
- **Age Groups**: Categorizes passengers into age groups
- **Fare Bins**: Creates fare categories for better model performance

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