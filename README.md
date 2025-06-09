# Housing Price Prediction System (XGBoost Regressor)

This project trains an advanced XGBoost regression model with hyperparameter tuning using `RandomizedSearchCV` to predict housing prices based on numerical features.

---

---

## Dataset
- **Boston Housing Dataset** (or any housing dataset)
- Features: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT
- Target: MEDV (Median value of owner-occupied homes)
- You can easily change them using pandas

> **Note**: You can also use your own housing dataset in `.csv` format with similar structure.

---

## Main Steps
1. Load and clean dataset (impute missing values)
2. Feature scaling using `StandardScaler`
3. Train/test split (default 75/25)
4. Hyperparameter tuning via `RandomizedSearchCV`
5. Final training using the best parameters
6. Evaluation (RMSE, R², MAE, MSE)
7. Plot feature importances

---

## How to Run

1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn xgboost
   ```

2. Run the script:
   ```bash
   python main.py
   ```

3. Provide the path to your dataset when prompted:
   ```
   Enter path to Housing CSV dataset: HousingData.csv
   ```

---

## Sample Output
```
Evaluating model on Test Set...
[Test Set] MSE=12.345, RMSE=3.514, R2=0.842, MAE=2.456
```

---

## Example CSV Format

| CRIM | ZN | INDUS | CHAS | NOX | RM | AGE | DIS | RAD | TAX | PTRATIO | B | LSTAT | MEDV |
|------|----|-------|------|-----|----|-----|-----|-----|-----|----------|---|--------|------|
| ...  | .. | ...   | ...  | ... | .. | ... | ... | ... | ... | ...      |...| ...    | ...  |

> Ensure the dataset includes target (label) column.

---

## Feature Importance Plot

The script will generate a horizontal bar chart showing the top features contributing to the prediction.

---

## Author

**alireza dehghanpour**
