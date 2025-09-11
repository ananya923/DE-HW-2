- Mini-assignment 2 for IDS706 Data Engineering (README made with the help if AI and edited manually for accuracy)
- Name: Ananya Jogalekar
- NetID: aj463

### Note about files

I created the repository by following the template provided in HW 1, hence I have files for Makefile, requirements, and testing. However, those were not expected in this assignment, so I haven't added any code to them. I might use them for the assignment next week, so they're still present in the repository for my future reference!

# Customer Behavior Analysis

This project explores customer purchase data to understand the drivers of Brand Loyalty.  
The workflow includes data cleaning, exploratory data analysis (EDA), predictive modeling, and visualization.

## Requirements
- pandas
- numpy
- train_test_split from sklearn.model_selection
- RandomForestRegressor from sklearn.ensemble
- mean_squared_error, r2_score from sklearn.metrics
- seaborn
- matplotlib.pyplot 

## Workflow

### 1. Data Import & Cleaning
- Loaded the dataset (`data.csv`) into a pandas DataFrame.
- Removed identifiers (`Customer_ID`) and cleaned column names.
- Cleaned numeric columns (e.g., `Purchase_Amount`).
- Encoded categorical variables using one-hot encoding for analysis.

### 2. Exploratory Data Analysis (EDA).
- Checked correlations between variables of interest, like Customer Satisfaction and Brand Loyalty.
- Identified Brand Loyalty as a key variable of interest for prediction.
- Defined my research question: "Identify the variable that is the most strongly correlated with Brand Loyalty."

### 3. Predictive Modeling
- Framed Brand Loyalty as a categorical target (e.g., low, medium, high).
- Applied Logistic Regression to predict customer loyalty.
- Evaluated model performance with accuracy and classification metrics.

### 4. Results
- Among all features, Age showed the strongest correlation with Brand Loyalty.

### 5. Visualization
- Used a Box Plot to illustrate the relationship between Age groups and Brand Loyalty.

## Tech Stack
- **Python** (pandas, scikit-learn, matplotlib, seaborn)
- **Machine Learning**: Logistic Regression
- **Visualization**: Box plots

## 6. Data Visualization Plot:
![alt text](image.png)