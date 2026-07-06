# 🏠 House Price Prediction — Linear Regression

A simple, interpretable linear regression model that predicts house sale prices using **square footage**, **bedroom count**, and **bathroom count**, built on the Ames Housing dataset (Kaggle's "House Prices: Advanced Regression Techniques").

---

## 📊 Project Overview

| | |
|---|---|
| **Goal** | Predict `SalePrice` for houses in the test set |
| **Model** | Linear Regression (scikit-learn) |
| **Features** | Total square footage, bedrooms, bathrooms |
| **Dataset** | Ames Housing (train.csv / test.csv) |
| **Validation R²** | ~0.72 |
| **Validation RMSE** | ~$46,300 |

---

## 📁 Project Structure

```
SCT_ML_Task1/
├── house_price_linear_regression.py   # Main script
├── train.csv                          # Training data (with SalePrice)
├── test.csv                           # Test data (no SalePrice — predict this)
├── sample_submission.csv              # Kaggle's expected submission format
├── submission.csv                     # Generated after running the script
└── README.md                          # This file
```

---

## 🧠 Features Used

The model was intentionally kept simple and interpretable, using three engineered feature groups:

1. **Total Square Footage** (`TotalSF`)
   `GrLivArea` (above-ground living area) + `TotalBsmtSF` (basement area)

2. **Bedrooms** (`Bedrooms`)
   `BedroomAbvGr` — bedrooms above grade

3. **Total Bathrooms** (`TotalBath`)
   `FullBath` + `0.5 × HalfBath` + `BsmtFullBath` + `0.5 × BsmtHalfBath`

---

## ⚙️ Setup & Installation

1. Clone or download this project folder.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn
   ```
3. Make sure `train.csv` and `test.csv` are in the same folder as the script (or update the file paths at the top of `house_price_linear_regression.py`).

---

## ▶️ How to Run

```bash
python house_price_linear_regression.py
```

The script will:
- Load and clean the data
- Engineer the three features above
- Split the training data (80/20) for validation
- Train a linear regression model
- Print evaluation metrics (MAE, RMSE, R², 5-fold CV)
- Show the learned coefficients and what they mean in plain English
- Retrain on the full dataset and generate `submission.csv`, ready for Kaggle

---

## 📈 Results

```
Validation MAE : $29,657
Validation RMSE: $46,299
Validation R²  : 0.7205
Mean 5-Fold CV R²: 0.6500
```

**Coefficients:**
| Feature | Effect on Price |
|---|---|
| Total Sq Ft | +$61 per sq ft |
| Bedrooms | −$12,741 per bedroom |
| Total Bathrooms | +$26,991 per bathroom |

> **Note on the negative bedroom coefficient:** This isn't a bug — it's a real (if counterintuitive) pattern in the data. When square footage is held constant, adding more bedrooms usually means each room is smaller, which can slightly reduce perceived value. This is a classic example of multicollinearity between features, and a good reminder to interpret linear regression coefficients carefully rather than assuming pure causation.

---

## 🚀 Possible Improvements

- Add more predictive features (`OverallQual`, `Neighborhood`, `YearBuilt` are strong predictors in this dataset)
- Try regularized models (Ridge, Lasso) to handle multicollinearity
- Try non-linear models (Random Forest, Gradient Boosting) for higher accuracy
- Log-transform `SalePrice` to reduce skew (common trick in this competition)

---

## 📄 License / Credit

Dataset: [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), compiled by Dean De Cock, distributed via Kaggle.
