"""
House Price Prediction — Linear Regression

Predicts SalePrice from square footage and bedroom/bathroom counts
using the Ames Housing dataset (Kaggle "House Prices" competition).

Features used:
  - Square footage : GrLivArea (above-grade living area) + TotalBsmtSF (basement area)
  - Bedrooms        : BedroomAbvGr
  - Bathrooms       : FullBath + HalfBath (0.5 weight) + BsmtFullBath + BsmtHalfBath (0.5 weight)

"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. LOAD DATA

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
OUTPUT_PATH = "submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")

# 2. FEATURE ENGINEERING

def build_features(df):
    """
    Build the 3 requested feature groups: square footage, bedrooms, bathrooms.
    NA basement/bath columns (test set only) mean 'no basement' -> fill with 0.
    """
    df = df.copy()

    # Fill NAs relevant to our features (only appear in test set, means "none")
    for col in ["TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Total square footage (above ground + basement) 
    df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]

    # Total bathrooms (half baths count as 0.5 of a full bath) 
    df["TotalBath"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    # Bedrooms 
    df["Bedrooms"] = df["BedroomAbvGr"]

    feature_cols = ["TotalSF", "Bedrooms", "TotalBath"]
    return df[feature_cols]


X = build_features(train_df)
y = train_df["SalePrice"]

X_kaggle_test = build_features(test_df)  # final unseen test set (no labels, for submission)

print("\nFeature preview:")
print(X.head())

# 3. TRAIN / VALIDATION SPLIT

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. TRAIN THE MODEL

model = LinearRegression()
model.fit(X_train, y_train)


# 5. EVALUATE

val_preds = model.predict(X_val)

mae = mean_absolute_error(y_val, val_preds)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
r2 = r2_score(y_val, val_preds)

print("\n--- Validation Performance ---")
print(f"MAE : ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R^2 : {r2:.4f}")

# 5-fold cross-validation on the full training set for a more robust estimate
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"\n5-Fold CV R^2 scores: {np.round(cv_scores, 4)}")
print(f"Mean CV R^2: {cv_scores.mean():.4f}")


# 6. INSPECT THE LEARNED MODEL

print("\n--- Model Coefficients ---")
for feat, coef in zip(X.columns, model.coef_):
    print(f"{feat:12s}: {coef:,.2f}")
print(f"{'Intercept':12s}: {model.intercept_:,.2f}")

print(
    "\nInterpretation:\n"
    f"  - Each additional sq ft adds ~${model.coef_[0]:,.2f} to predicted price\n"
    f"  - Each additional bedroom changes price by ~${model.coef_[1]:,.2f}\n"
    f"  - Each additional bathroom adds ~${model.coef_[2]:,.2f} to predicted price"
)

# 7. RETRAIN ON FULL TRAINING DATA

final_model = LinearRegression()
final_model.fit(X, y)

test_predictions = final_model.predict(X_kaggle_test)
test_predictions = np.clip(test_predictions, a_min=0, a_max=None)  # no negative prices

submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": test_predictions
})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission file saved to: {OUTPUT_PATH}")
print(submission.head())


# 8. EXAMPLE

example = pd.DataFrame({
    "TotalSF": [2500],
    "Bedrooms": [3],
    "TotalBath": [2.5]
})
example_pred = final_model.predict(example)[0]
print(f"\nExample: A 2500 sq ft house, 3 bedrooms, 2.5 bathrooms")
print(f"Predicted price: ${example_pred:,.2f}")
