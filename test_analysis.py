# Name: Ananya Jogalekar
# NetID: aj463
# Test File
# Note on AI use: AI was used for suggestions on which code
# to test and which parts to leave out. It was also used to debug
# the syntax for assert statements.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import unittest

# Refactoring previous code to create testing functions
def load_clean_data(filepath):    # Can specify dtypes as : str) -> pd.DataFrame:
    data = pd.read_csv(filepath)
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

# Unit Testing
class TestAnalysis(unittest.TestCase):

    def setUp(self):
        # create a small mock dataset for testing
        # We can edit to add custom test cases with the code below
        self.mock_data = pd.DataFrame({
            "Customer_ID": [1, 2, 3, 4],
            "Time_Spent_on_Product_Research(hours)": [2, 3, 4, 5],
            "Purchase_Amount": ["$100", "$200", "$150", "$300"],
            "Time_of_Purchase": ["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01"],
            "Income_Level": ["Low", "Medium", "High", "Medium"],
            "Social_Media_Influence": ["Yes", "No", "Yes", "No"],
            "Purchase_Intent": ["High", "Low", "Medium", "High"],
            "Age": [25, 35, 45, 55],
            "Brand_Loyalty": [3, 4, 2, 5]
        })
        self.mock_path = "mock_data.csv" # Make sure we don't overwrite original data file!
        self.mock_data.to_csv(self.mock_path, index=False)

    def test_load_clean_data(self):
        df = load_clean_data(self.mock_path)
        
        # test that columns are renamed
        self.assertIn("Time_Spent_on_Product_Research", df.columns)
        
        # test that Purchase_Amount is numeric
        self.assertTrue(pd.api.types.is_float_dtype(df["Purchase_Amount"]))
        
        # test that Time_of_Purchase is datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["Time_of_Purchase"]))

    def test_train_random_forest(self):
        df = load_clean_data(self.mock_path)
        model, mse, r2 = train_random_forest(df)
        
        # check that model is trained
        self.assertTrue(hasattr(model, "predict"))
        
        # check mse and r2 are floats
        self.assertIsInstance(mse, float)
        self.assertIsInstance(r2, float)

if __name__ == '__main__':
    unittest.main()

