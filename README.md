Mini-assignment 2 for IDS706 Data Engineering (README made with the help if AI and edited manually for accuracy)
Name: Ananya Jogalekar
NetID: aj463

# Customer Behavior Analysis

This project explores customer purchase data to understand the drivers of Brand Loyalty.  
The workflow includes data cleaning, exploratory data analysis (EDA), predictive modeling, and visualization.

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