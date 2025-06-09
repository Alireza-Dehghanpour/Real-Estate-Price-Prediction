# main.py
# alireza dehghanpour

"""
Housing Price Prediction System - Advanced Version
*****************************************************
This script trains an XGBoost regression model with hyperparameter tuning 
to predict housing prices based on numerical features (e.g., you can use the Boston Housing Dataset as sample).

Main Steps:
1. Load and clean dataset (impute missing values).
2. Scale numerical features.
3. Hyperparameter tuning using RandomizedSearchCV.
4. Train XGBoostRegressor with best parameters.
5. Evaluate the model on train and test sets.
6. Plot feature importances.
"""

#...............................Libraries................................
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from xgboost import XGBRegressor

#...............................Functions................................

def load_data(filepath):
    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape[0]} rows {df.shape[1]} columns")
    missing_before = df.isnull().sum().sum()
    df.fillna(df.mean(), inplace=True)
    missing_after = df.isnull().sum().sum()
    print(f"Missing values: before={missing_before}, after={missing_after}")
    return df



def preprocess_data(df, target_col='MEDV'):
    print("Preprocessing data...")
    X = df.drop(columns=target_col)
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Preprocessing completed.")
    return X_scaled, y, scaler, X.columns



def split_data(X, y, test_size=0.25, random_state=42):
    print("Splitting data into training and test sets...")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)




def tune_hyperparameters(X_train, y_train):
    print("Tuning hyperparameters using RandomizedSearchCV...")
    param_dist = {
        'n_estimators':      [50, 100, 150, 200],
        'max_depth':         [3, 5, 7, 9],
        'learning_rate':     [0.01, 0.05, 0.1, 0.2],
        'subsample':         [0.6, 0.8, 1.0],
        'colsample_bytree':  [0.6, 0.8, 1.0],
        'gamma':             [0, 0.1, 0.2, 0.5],
        'min_child_weight':  [1, 3, 5]
    }



    base_model = XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=50,
        scoring='neg_mean_squared_error',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    print("Best hyperparameters:", search.best_params_)
    return search.best_params_




def train_model(X_train, y_train, best_params):
    print("Training final XGBoostRegressor model...")
    model = XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    model.fit(X_train, y_train)
    return model



def evaluate_model(model, X, y, dataset_name="Dataset"):
    print(f"\nEvaluating model on {dataset_name}...")
    preds = model.predict(X)
    mse  = metrics.mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    r2   = metrics.r2_score(y, preds)
    mae  = metrics.mean_absolute_error(y, preds)
    print(f"[{dataset_name}] MSE={mse:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, MAE={mae:.3f}")
    
    

def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(8,6))
    plt.barh(indices, importances[indices])
    plt.yticks(indices, feature_names[indices])
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

#...............................Main................................

if __name__ == "__main__":
    print("\nHousing Price Prediction System - XGBoost Regressor")
    print("-" * 55)
    filepath = input("Enter path to Housing CSV dataset: ").strip()
    df = load_data(filepath)
    X_scaled, y, scaler, feature_names = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    best_params = tune_hyperparameters(X_train, y_train)
    model = train_model(X_train, y_train, best_params)

    evaluate_model(model, X_train, y_train, "Training Set")
    evaluate_model(model, X_test,  y_test,  "Test Set")
    plot_feature_importances(model, feature_names)
