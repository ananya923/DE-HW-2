# Name: Ananya Jogalekar
# NetID: aj463
# Test File
# Note on AI use: AI was used debug the syntax for assert statements.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# importing the functions we want to test from other files
from refactored_analysis import load_data, clean_data, explore_data, train_random_forest


# Testing
def test_loading_data():
    df = load_data("data.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1000, 28)

def test_cleaning_data():
    df = load_data("data.csv")
    df = clean_data(df)

    # test that columns are renamed
    assert "Time_Spent_on_Product_Research" in df.columns

    # test that Purchase_Amount is numeric
    assert pd.api.types.is_float_dtype(df["Purchase_Amount"])

    # test that Time_of_Purchase is datetime
    assert pd.api.types.is_datetime64_any_dtype(df["Time_of_Purchase"])
    
    # test for Age col: values should be within range 18 to 100
    assert df["Age"].between(18, 100).all()

    # test for Brand Loyalty col: values should be within range 1 to 5
    assert df["Brand_Loyalty"].between(1, 5).all()

    # test for Cust Satisfacion col: values should be within range 1 to 10
    assert df["Customer_Satisfaction"].between(1, 10).all()

    # test for Purchase Channel col: values should be either Online, In-store, or Mixed
    valid_channels = {"Online", "In-Store", "Mixed"}
    assert set(df["Purchase_Channel"].unique()).issubset(valid_channels)

    # # EDGE CASES ##

    # check for duplicate Cust IDs
    assert df["Customer_ID"].nunique() == len(df)

    # there should be no negative values in fields like frequency and time
    for col in ["Frequency_of_Purchase", "Return_Rate", "Time_to_Decision"]:
        assert (df[col] >= 0).all()

    # TEST for ML reliability: target variable shouldn't be too skewed.
    # This is a categorical field, so no single category should
    # have a very high frequency in the raw data.
    value_counts = df["Brand_Loyalty"].value_counts(normalize=True)
    assert (value_counts < 0.9).all()

def test_exploring_data():
    df = load_data("data.csv")
    df = clean_data(df)

    results = explore_data(df)

    # Test that all metrics are returned as expected
    expected_keys = {"loyalty_income", "loyalty_researchTime", "loyalty_mediaInfluence",
        "loyalty_intent", "satis_income", "satis_researchTime",
        "satis_mediaInfluence", "satis_intent"}
    
    assert set(results.keys()) == expected_keys

    # Testing summary stats for Brand Loyalty
    # group by Income_Level
    assert isinstance(results['loyalty_income'], pd.DataFrame)
    assert "mean" in results['loyalty_income'].columns
    assert results['loyalty_income'].index.isin(df["Income_Level"].unique()).all()

    # Correlation with research time
    assert isinstance(results["loyalty_researchTime"], float)
    assert -1 <= results["loyalty_researchTime"] <= 1

    # # Group by Social Media Influence
    assert isinstance(results['loyalty_mediaInfluence'], pd.DataFrame)
    assert not results['loyalty_mediaInfluence'].empty

    # # Group by Purchase Intent
    assert isinstance(results['loyalty_intent'], pd.DataFrame)
    assert not results['loyalty_intent'].empty

    # # Testing summary stats for Customer Satisfatction
    # Group by Income_Level
    assert isinstance(results['satis_income'], pd.DataFrame)
    assert "mean" in results['satis_income'].columns
    assert results['satis_income'].index.isin(df["Income_Level"].unique()).all()

    # # Correlation with research time
    assert isinstance(results['satis_researchTime'], float)
    assert -1 <= results['satis_researchTime'] <= 1

    # # Group by Social Media Influence
    assert isinstance(results['satis_mediaInfluence'], pd.DataFrame)
    assert not results['satis_mediaInfluence'].empty

    # # Group by Purchase Intent
    assert isinstance(results['satis_intent'], pd.DataFrame)
    assert not results['satis_intent'].empty


# # testing the ML model ##
def test_train_random_forest():
    df = load_data("data.csv")
    df = clean_data(df)
    model, mse, r2 = train_random_forest(df)

    # model should have predict method
    assert hasattr(model, "predict")

    # metrics should be floats
    assert isinstance(mse, float)
    assert isinstance(r2, float)



