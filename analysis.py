# Name: Ananya Sameer Jogalekar
# NetID: aj463
# IDS 706 Data Engineering Week 2 Mini-Assignment - First Major Assignment
# Dataset chosen: 'Ecommerce Consumer Behavior Analysis Data' from suggested datasets

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a filepath and return a pandas DataFrame.
    """
    customer_data = pd.read_csv(filepath)
    return customer_data


def clean_data(customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by renaming columns, handling missing values,
    converting data types, and parsing dates.
    """
    customer_data = customer_data.rename(
        columns={
            "Time_Spent_on_Product_Research(hours)": "Time_Spent_on_Product_Research"
        }
    )

    # Drop missing values only in relevant columns to avoid losing too much data
    customer_data = customer_data.dropna(
        subset=["Purchase_Amount", "Time_of_Purchase", "Time_Spent_on_Product_Research"]
    )

    customer_data["Purchase_Amount"] = (
        customer_data["Purchase_Amount"].str.replace("$", "", regex=False).astype(float)
    )

    customer_data["Time_of_Purchase"] = pd.to_datetime(
        customer_data["Time_of_Purchase"]
    )
    return customer_data


# New Helper Functions: Refactored to remove redundancy
def groupby_income(data: pd.DataFrame, target_variable):
    """
    Compare any target variable to Income_Level to see if there is a correlation.
    Running groupby on high volume of data can be slow. However, income is an important
    variable to consider when analyzing customer behavior.
    """
    var_by_income = data.groupby("Income_Level")[str(target_variable)].describe()
    print(var_by_income)
    return var_by_income


def corr_by_researchTime(data: pd.DataFrame, target_variable):
    """
    Compare any target variable to Time_Spent_on_Product_Research to see if there is a correlation.
    """
    var_by_researchTime = data["Time_Spent_on_Product_Research"].corr(
        data[str(target_variable)]
    )
    print(round(var_by_researchTime, 2))
    return var_by_researchTime


def groupby_mediaInfluence(data: pd.DataFrame, target_variable):
    """
    Compare any target variable to Social_Media_Influence to see if there is a correlation.
    Running groupby on high volume of data can be slow. However, this dataset is about marketing
    and consumer sentiments. Hence, social media influence is important
    for the domain we are dealing with.
    """
    var_by_mediaInfluence = data.groupby("Social_Media_Influence")[
        str(target_variable)
    ].describe()
    print(var_by_mediaInfluence)
    return var_by_mediaInfluence


def groupby_purchaseIntent(data: pd.DataFrame, target_variable):
    """
    Compare any target variable to Purchase_Intent to see if there is a correlation.
    Running groupby on high volume of data can be slow. However, like social media influence,
    purchase intent is important for market research.
    """
    var_by_intent = data.groupby("Purchase_Intent")[str(target_variable)].describe()
    print(var_by_intent)
    return var_by_intent


def explore_data(customer_data):
    """
    Perform exploratory data analysis and return summary statistics.
    """

    # Summary Stats For Brand Loyalty
    print("Summary Stats For Brand Loyalty:")
    loyalty_by_income = groupby_income(customer_data, "Brand_Loyalty")

    loyalty_by_researchTime = corr_by_researchTime(customer_data, "Brand_Loyalty")

    loyalty_by_mediaInfluence = groupby_mediaInfluence(customer_data, "Brand_Loyalty")

    loyalty_by_intent = groupby_purchaseIntent(customer_data, "Brand_Loyalty")

    # Summary Stats For Customer Satisfaction
    print("Summary Stats For Customer Satisfaction:")
    satis_by_income = groupby_income(customer_data, "Customer_Satisfaction")

    satis_by_researchTime = corr_by_researchTime(customer_data, "Customer_Satisfaction")

    satis_by_mediaInfluence = groupby_mediaInfluence(
        customer_data, "Customer_Satisfaction"
    )

    satis_by_intent = groupby_purchaseIntent(customer_data, "Customer_Satisfaction")

    return {
        "loyalty_by_income": loyalty_by_income,
        "loyalty_by_researchTime": loyalty_by_researchTime,
        "loyalty_by_mediaInfluence": loyalty_by_mediaInfluence,
        "loyalty_by_intent": loyalty_by_intent,
        "satis_by_income": satis_by_income,
        "satis_by_researchTime": satis_by_researchTime,
        "satis_by_mediaInfluence": satis_by_mediaInfluence,
        "satis_by_intent": satis_by_intent,
    }


def run_random_forest(customer_data, target="Brand_Loyalty"):
    """
    Run a Random Forest regressor to predict any target variable. By deafult,
    the target variable is set to "Brand_Loyalty", because that appears to be the
    most relevant outcome that marketing teams would want to achieve and predict.
    """
    # Making train and variable sets
    X = customer_data.drop(columns=["Customer_ID", "Time_of_Purchase", target])
    y = customer_data[target]

    # Specifying train and test datasets
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # Fitting the random forest model
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluating the model
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rf, mse, r2


# Data Visualization
def create_visualization(data_file_path: str):
    print("Visualizing Age against Brand Loyalty:")
    customer_data = load_data(filepath=data_file_path)
    my_chart = sns.boxplot(data=customer_data, x="Brand_Loyalty", y="Age")
    my_chart.set_title("Relationship between Age and Brand Loyalty")
    plt.show()


# Calling the data viz function to render the plot
create_visualization("data.csv")
