import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Refactoring previous code to create testing functions
def load_data(filepath):  # Can specify dtypes as : str) -> pd.DataFrame:
    data = pd.read_csv(filepath)  # this should return a proper df with all cols
    return data  # we will test that in test_analysis.py


def clean_data(data):
    data = data.rename(
        columns={
            "Time_Spent_on_Product_Research(hours)": "Time_Spent_on_Product_Research"
        }
    )
    data = data.dropna()
    data["Purchase_Amount"] = (
        data["Purchase_Amount"].str.replace("$", "", regex=False).astype(float)
    )
    data["Time_of_Purchase"] = pd.to_datetime(data["Time_of_Purchase"])
    return data


def explore_data(data):
    # Summary Stats For Brand Loyalty
    print("Summary Stats For Brand Loyalty:")
    loyalty_income = data.groupby("Income_Level")["Brand_Loyalty"].describe()
    print(loyalty_income)

    loyalty_researchTime = data["Time_Spent_on_Product_Research"].corr(
        data["Brand_Loyalty"]
    )
    print(round(loyalty_researchTime, 2))

    loyalty_mediaInfluence = data.groupby("Social_Media_Influence")[
        "Brand_Loyalty"
    ].describe()
    print(loyalty_mediaInfluence)

    loyalty_intent = data.groupby("Purchase_Intent")["Brand_Loyalty"].describe()
    print(loyalty_intent)

    # Summary Stats For Customer Satisfaction
    print("Summary Stats For Customer Satisfaction:")
    satis_income = data.groupby("Income_Level")["Customer_Satisfaction"].describe()
    print(satis_income)

    satis_researchTime = data["Time_Spent_on_Product_Research"].corr(
        data["Customer_Satisfaction"]
    )
    print(round(satis_researchTime, 2))

    satis_mediaInfluence = data.groupby("Social_Media_Influence")[
        "Customer_Satisfaction"
    ].describe()
    print(satis_mediaInfluence)

    satis_intent = data.groupby("Purchase_Intent")["Social_Media_Influence"].describe()
    print(satis_intent)

    return {
        "loyalty_income": loyalty_income,
        "loyalty_researchTime": loyalty_researchTime,
        "loyalty_mediaInfluence": loyalty_mediaInfluence,
        "loyalty_intent": loyalty_intent,
        "satis_income": satis_income,
        "satis_researchTime": satis_researchTime,
        "satis_mediaInfluence": satis_mediaInfluence,
        "satis_intent": satis_intent,
    }


def train_random_forest(data, target="Brand_Loyalty"):
    X = data.drop(columns=["Customer_ID", "Time_of_Purchase", target])
    y = data[target]
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rf, mse, r2
