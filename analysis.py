# Name: Ananya Sameer Jogalekar
# NetID: aj463
# IDS 706 Data Engineering Week 2 Mini-Assignment: Start Your First Data Analysis
# Dataset chosen: 'Ecommerce Consumer Behavior Analysis Data' from suggested datasets

# Importing necessary libraries
import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes # I installed kmodes after getting a ModuleNotFoundError on my machine

# Import the Dataset
data = pd.read_csv('Data.csv')

# Inspect the Data
# E.g., Display the first few rows using .head() to get a quick overview.
# E.g., Use .info() and .describe() to understand data types and summary statistics.
# E.g., Check for missing values and duplicates (optional).
print(data.head())
print('Number of columns = ', len(data.columns), '\n', data.columns)

data.info()
# here we notice some missing values in cols Social_Media_Influence and Engagement_with_Ads
# we note this for dealing with missing values later

# inspect data types, range of values, categories
data.describe()     # most vars seem to have reasonable measures of central tendency
data.nunique()      
# many variables seem to be categorical. We need to know what the categories are.

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

# Reference for Analysis:
# Independent Vars: Age, Gender, Income, Marital Status, Education, Occupation
#       Time on Product Research, Device used, Customer Loyalty Program member, Social Media Influence,
#       Engagement with Ads, Payment method, Time of Purchase, Discount Used, Purchase Intent, Shipping Preference
# Dependent Vars: Purchase category, Purchase amt, Frequency of Purchase,
#       Brand Loyalty, Product Rating, Return rate, customer satisfaction
# Vars on the fence: Purchase Channelz, Discount sensitivity, Time to Decision


# Basic Filtering and Grouping
# Apply filters to extract meaningful subsets of the data.
# Use groupby() or equivalent on a selected variable and compute summary statistics (e.g., mean, count).


# Aim: find vars that correlate strongly with Brand Loyalty and Customer Satisfaction

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

# Using the K-Prototypes algorithm for customer segmentation
# across categorical as well as numeric variables.

# Separate categorical and numeric columns
categorical_cols = ['Income_Level', 'Social_Media_Influence', 'Purchase_Intent']
numeric_cols = ['Brand_Loyalty', 'Customer_Satisfaction']

# Convert to numpy array
matrix = data[categorical_cols + numeric_cols].to_numpy().astype(object)

# Fit K-Prototypes
kproto = KPrototypes(n_clusters=4, init='Cao', verbose=2)
clusters = kproto.fit_predict(matrix, categorical=[0, 1])

# Assign cluster labels
data['Cluster'] = clusters






# Visualization
# Create one plot (e.g., histogram, boxplot, scatter plot) using Matplotlib, Seaborn, or others.


# Documentation
# Explain your steps and findings in a README.md file.