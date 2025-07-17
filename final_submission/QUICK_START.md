# Titanic Survival Predictor - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Data
Place the Titanic dataset files in a `titanic/` directory:
- `titanic/train.csv` - Training data
- `titanic/test.csv` - Test data

### Step 3: Run the Pipeline
```bash
# Run complete ML pipeline
python main.py pipeline --verbose
```

This will automatically:
1. ✅ Explore and analyze the data
2. ✅ Train multiple models with hyperparameter tuning
3. ✅ Select the best performing model (~84% accuracy)
4. ✅ Generate predictions for test set
5. ✅ Create submission file at `outputs/predictions/submission.csv`

## 📊 Expected Results

- **Model Performance**: ~84% accuracy, ~85% ROC AUC
- **Submission File**: Ready for Kaggle upload
- **Visualizations**: Generated in `outputs/visualizations/`
- **Detailed Logs**: Available in `titanic_predictor.log`

## 🧪 Verify Installation

```bash
# Run test suite (187 tests should pass)
python -m pytest tests/ -v
```

## 🔧 Individual Commands

```bash
# Data exploration only
python main.py explore --verbose

# Train models only
python main.py train --verbose

# Generate predictions only
python main.py predict --verbose

# Evaluate model performance
python main.py evaluate --verbose
```

## 📁 Output Structure

After running, you'll have:
```
outputs/
├── models/                 # Trained models
├── predictions/           # submission.csv for Kaggle
└── visualizations/        # Generated plots
```

## ❓ Need Help?

- Check `README.md` for detailed documentation
- Review `titanic_predictor.log` for execution details
- All 187 tests should pass with `pytest tests/`

**Ready to predict Titanic survival? Just run the pipeline command above!**