import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures

import mlflow

df = pd.read_csv("train.csv")

numeric_df = df.select_dtypes(include='number')  # Select only numeric columns
correlations = numeric_df.corr()  # Calculate correlations
important_num_cols = list(correlations["SalePrice"][(correlations["SalePrice"] > 0.50) | (correlations["SalePrice"] < -0.50)].index)
cat_cols = ["MSZoning", "Utilities", "BldgType", "Heating", "KitchenQual", "SaleCondition", "LandSlope"]
important_cols = important_num_cols + cat_cols

df = df[important_cols]
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]
X = pd.get_dummies(X, columns=cat_cols)
important_num_cols.remove("SalePrice")

scaler = StandardScaler()
X[important_num_cols] = scaler.fit_transform(X[important_num_cols])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5)).mean()
    return rmse
    

def evaluation(y, predictions):
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r_squared = r2_score(y, predictions)
    return mae, mse, rmse, r_squared

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("House Prices Prediction1")
with mlflow.start_run(run_name="run_1"):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    model_info = mlflow.sklearn.log_model(lin_reg,"house-price-model")
    predictions = lin_reg.predict(X_test)
    
    mae, mse, rmse, r_squared = evaluation(y_test, predictions)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r_squared)
    print("-"*30)
    rmse_cross_val = rmse_cv(lin_reg)
    print("RMSE Cross-Validation:", rmse_cross_val)
    
    new_row = {"Model": "LinearRegression","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared, "RMSE (Cross-Validation)": rmse_cross_val}
    #models = models.append(new_row, ignore_index=True)
    metric_eval = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared}
    # Log the loss metric
    mlflow.log_metrics(metric_eval)
    
    # Log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=lin_reg,
        artifact_path="sklearn-model",
        registered_model_name="sk-learn-lin-reg-model",
    )