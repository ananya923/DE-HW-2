# Name: Ananya Sameer Jogalekar
# NetID: aj463
# IDS 706 Data Engineering Week 2 Mini-Assignment: Start Your First Data Analysis
# Dataset chosen: 'Ecommerce Consumer Behavior Analysis Data' from suggested datasets

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Import the Dataset
data = pd.read_csv('Data.csv')

# Inspect the Data
print(data.head())
print('Number of columns = ', len(data.columns), '\n', data.columns)

data.info()
data.describe()     # most vars seem to have reasonable measures of central tendency
data.nunique()      
# many variables seem to be categorical. We need to know what the categories are.

# clean col names
data = data.rename(columns={'Time_Spent_on_Product_Research(hours)':'Time_Spent_on_Product_Research'})

# drop missing values
data = data.dropna()

# Checking non-numeric variables
cols_list = list(data.columns)
cols_list.remove('Customer_ID') # we don't need to check unique values for this

# see unique values taken by every mon-numeric variable
for col in cols_list:
    if data[col].dtype == 'object' or data[col].dtype == 'bool':
        print(col, ' : ', data[col].unique())
# we notice that Purchase_Amount and Time_of_Purchase 
# are non-numeric even though they describe quantifiable variables.
# So maybe we shouldn't include them when we think of categorical vars.

# convert the problematic cols into numeric dtypes
# currency column
data['Purchase_Amount'] = data['Purchase_Amount'].str.replace('$', '', regex=False)
data['Purchase_Amount'] = data['Purchase_Amount'].astype(float)

# date column
data['Time_of_Purchase'] = pd.to_datetime(data['Time_of_Purchase'])


# Basic Filtering and Grouping
# Apply filters to extract meaningful subsets of the data.
# Use groupby() or equivalent on a selected variable and compute summary statistics (e.g., mean, count).

# Summary Stats For Brand Loyalty
loyalty_income = data.groupby('Income_Level')['Brand_Loyalty'].describe()
print(loyalty_income)

loyalty_researchTime = data['Time_Spent_on_Product_Research(hours)'].corr(data['Brand_Loyalty'])
print(round(loyalty_researchTime,2))

loyalty_mediaInfluence = data.groupby('Social_Media_Influence')['Brand_Loyalty'].describe()
print(loyalty_mediaInfluence) # this shows some variation across levels of influence

loyalty_intent = data.groupby('Purchase_Intent')['Brand_Loyalty'].describe()
print(loyalty_intent)


# Summary Stats For Customer Satisfaction
satis_income = data.groupby('Income_Level')['Customer_Satisfaction'].describe()
print(satis_income)

satis_researchTime = data['Time_Spent_on_Product_Research(hours)'].corr(data['Customer_Satisfaction'])
print(round(satis_researchTime,2))

satis_mediaInfluence = data.groupby('Social_Media_Influence')['Customer_Satisfaction'].describe()
print(satis_mediaInfluence) # this shows some variation across levels of influence

satis_intent = data.groupby('Purchase_Intent')['Social_Media_Influence'].describe()
print(satis_intent)


# Explore a Machine Learning Algorithm
# Choose an ML algorithm.
# Begin experimenting with model inputs and outputs.

target = "Brand_Loyalty"
X = data.drop(columns=["Customer_ID", target])
y = data[target]

# Encoding categorical variables to prepare them for ML algorithm
X_encoded = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Random Forest model
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Regressor")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X_encoded.columns).sort_values(ascending=False)
print("\nTop Features Driving Brand Loyalty:")
print(importances.head(10))

# Age comes out as the top factor correlated with Brand Loyalty.
# It'd be nice to see a plot for more details.

# Visualization
# Create one plot (e.g., histogram, boxplot, scatter plot) using Matplotlib, Seaborn, or others.
sns.boxplot(data=data, x="Brand_Loyalty", y="Age")
plt.show()

# The plot is not very clear as the correlation coefficient is quite low
# but we observe slight differences across loyalty levels.