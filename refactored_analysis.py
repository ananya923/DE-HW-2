import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Refactoring previous code to create testing functions
def load_data(filepath):    # Can specify dtypes as : str) -> pd.DataFrame:
    data = pd.read_csv(filepath)
    return data

def clean_data(data):
    data = data.rename(columns={'Time_Spent_on_Product_Research(hours)':'Time_Spent_on_Product_Research'})
    data = data.dropna()
    data['Purchase_Amount'] = data['Purchase_Amount'].str.replace('$', '', regex=False).astype(float)
    data['Time_of_Purchase'] = pd.to_datetime(data['Time_of_Purchase'])
    return data

def train_random_forest(data, target = "Brand_Loyalty"):
    X = data.drop(columns=["Customer_ID", "Time_of_Purchase", target])
    y = data[target]
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rf, mse, r2
