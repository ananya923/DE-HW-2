- Mini-assignment 2 for IDS706 Data Engineering (README made with the help if AI and edited manually for accuracy)
- Name: Ananya Jogalekar
- NetID: aj463

# Project Description: Customer Behavior Analysis

This project explores customer purchase data to understand the drivers of Brand Loyalty.  
The workflow includes data cleaning, exploratory data analysis (EDA), predictive modeling, visualization, and testing.

## Requirements
- pandas
- numpy
- train_test_split from sklearn.model_selection
- RandomForestRegressor from sklearn.ensemble
- mean_squared_error, r2_score from sklearn.metrics
- seaborn
- matplotlib.pyplot 

# Environment Setup Instructions
## Devcontainer Setup
This project includes a `.devcontainer` configuration for a consistent development environment in VS Code. It can be enabled using the following steps:
1. Open your repository in VS Code.
2. Install the **Dev Containers** extension from the Extensions tab on the left.
3. Reopen the project in the container by clicking on the blue icon in the bottom left corner (choose the option `Remote-Containers: Reopen in Container`).
4. All dependencies will be installed automatically, ensuring reproducibility. To cross-check, make sure that the newly opened environment has a devcontainer in its title.
5. If working locally in VS Code, push all of your changes to GitHub. A new folder called `.devcontainer` would be created in your repository.

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

## 6. Data Visualization Plot:
![alt text](image.png)

# 7. Testing
- Testing is implemented with **pytest** to ensure correctness and reliability of the pipeline.  

### Test Cases

1. **test_loading_data**
   - Ensures the dataset is loaded correctly into a pandas DataFrame.
   - Verifies the shape of the DataFrame matches expectations (1000 rows × 28 columns).

2. **test_cleaning_data**
   - Validates that cleaning functions properly transform the dataset:
     - Column renaming is applied correctly.
     - `Purchase_Amount` is converted to numeric.
     - `Time_of_Purchase` is converted to datetime.
     - `Age` values lie within a valid range (18–100).
     - `Brand_Loyalty` values are between 1–5.
     - `Customer_Satisfaction` values are between 1–10.
     - `Purchase_Channel` contains only valid categories (Online, In-Store, Mixed).
   - Edge case checks:
     - No duplicate `Customer_ID`s.
     - No negative values in fields like frequency, return rate, or decision time.
     - Target variable (`Brand_Loyalty`) distribution is not overly skewed.

3. **test_exploring_data**
   - Confirms that the `explore_data` function returns all expected metrics in a dictionary.
   - Validates the contents of each returned object:
     - Grouped statistics (`loyalty_income`, `loyalty_mediaInfluence`, etc.) are DataFrames with expected structure (contain means, correct indices, non-empty).
     - Correlation outputs (`loyalty_researchTime`, `satis_researchTime`) are floats between -1 and 1.

4. **test_train_random_forest**
   - Checks that the Random Forest model is trained correctly:
     - Returned model has a `.predict` method.
     - Performance metrics (MSE, R²) are floats.

# 8. File & Repository Structure
```
├── .devcontainer/ # Devcontainer configuration files
├── pycache/ # Compiled Python cache files
├── Makefile # Commands for building, testing, automation
├── README.md # Project documentation
├── analysis.py # Initial analysis script
├── data.csv # Dataset (raw input file)
├── refactored_analysis.py # Refactored analysis functions (load, clean, explore, train)
├── requirements.txt # Python dependencies
├── test_analysis.py # Pytest test cases
```
### Tech Stack
- **Python** (pandas, scikit-learn, matplotlib, seaborn)
- **Machine Learning**: Logistic Regression
- **Visualization**: Box plots