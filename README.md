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
pip install numpy pandas scikit-learn matplotlib seaborn pytest joblib xgboost lightgbm
```

### Installation

1. Clone or download the project
2. Ensure you have the Titanic dataset files in the `titanic/` directory:
   - `titanic/train.csv`
   - `titanic/test.csv`

### 🎯 For Beginners: Understanding the Pipeline

If you're new to data science, here's how this project works step-by-step:

#### **Step 1: Data Understanding** 
```bash
# Look at the raw data first
python -c "import pandas as pd; print(pd.read_csv('titanic/train.csv').head())"
```
The Titanic dataset contains information about passengers like age, sex, ticket class, family size, etc. Our goal is to predict who survived based on these features.

#### **Step 2: Run the Best Model (Beginner-Friendly)**
```bash
# Run our top-performing approach
python main_mvp_rules.py
```
This creates `mvp_improved_submission.csv` with 80.14% accuracy - our best result!

#### **Step 3: Understanding What Happened**
The magic happens in two parts:
1. **Smart Rules**: We found 18 passengers where family survival patterns give us high confidence
2. **Machine Learning**: For everyone else, we use XGBoost to make predictions

### 🔍 How the Code Works (Beginner Explanation)

#### **The Data Science Process**
```
Raw Data → Clean Data → Engineer Features → Train Model → Make Predictions
```

**1. Data Cleaning (`src/data/preprocessor.py`)**
- Fill missing ages using passenger titles (Master = child, Mr = adult man, etc.)
- Handle missing cabin and fare information
- Convert text data to numbers that computers can understand

**2. Feature Engineering (Making Data Smarter)**
- Extract titles from names: "Smith, Mr. John" → "Mr"
- Calculate family size: SibSp + Parch + 1
- Create interaction features: Age × Class, Age × Fare
- Extract deck information from cabin numbers

**3. The Breakthrough: Family Survival Rules**
```python
# Rule 1: If a Master's family all survived → He survives
# Rule 2: If a female's family all died → She dies
# These 18 high-confidence predictions boost our score significantly!
```

**4. Machine Learning for the Rest**
- Use XGBoost (a powerful algorithm) for remaining 400 passengers
- XGBoost learns patterns like: "1st class females usually survive"
- Combine rule predictions with ML predictions

### 📚 Learning Path for Beginners

#### **Level 1: Start Simple**
```bash
# Run the basic pipeline first
python main.py pipeline --verbose
```
This uses traditional ML approaches (Random Forest, Logistic Regression) with standard feature engineering.

#### **Level 2: Advanced Features**
```bash
# Try the enhanced approach with 40+ features
python main_enhanced.py
```
This adds sophisticated feature engineering like interaction terms and ensemble methods.

#### **Level 3: Domain Knowledge**
```bash
# Run the breakthrough approach
python main_mvp_rules.py
```
This combines domain-specific rules with ML - our top 4% solution!

### 🧠 Understanding Different Approaches

| File | Approach | Score | What You'll Learn |
|------|----------|-------|-------------------|
| `main.py` | Traditional ML | 0.76794 | Basic data science pipeline |
| `main_enhanced.py` | Advanced ML | 0.77033 | Feature engineering & ensembles |
| `main_mvp_rules.py` | Rules + ML | 0.80143 | Domain knowledge integration |

### 🔍 Code Structure Explained

**Core Components:**
- `src/data/loader.py` - Loads and validates the CSV files
- `src/data/preprocessor.py` - Cleans data and creates features
- `src/models/trainer.py` - Trains multiple ML algorithms
- `src/models/evaluator.py` - Tests model performance
- `src/models/predictor.py` - Makes final predictions

**Advanced Components:**
- `src/data/enhanced_preprocessor.py` - 40+ advanced features
- `main_mvp_rules.py` - Family survival rules implementation
- `compare_models.py` - Compares different approaches

### 💡 Key Insights for Data Scientists

1. **Domain Knowledge Beats Complex ML**: Simple family rules outperformed sophisticated ensembles
2. **Feature Engineering Matters**: Going from 12 to 40+ features improved accuracy significantly  
3. **Start Simple, Then Optimize**: Basic gender rule (females live, males die) gets you 82.3% baseline
4. **Validation is Critical**: Cross-validation scores helped select the best approach

### Running the Complete Pipeline

```bash
# Run the entire ML pipeline (original approach)
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