[![CICD for Major Assessment on Refactoring](https://github.com/ananya923/DE-HW-2/actions/workflows/CI.yml/badge.svg)](https://github.com/ananya923/DE-HW-2/actions/workflows/CI.yml)

# Consumer Behavior Analysis: Exploring Data and Predicting Brand Loyalty

## 1. Overview

This project explores customer purchase data to understand the drivers of Brand Loyalty. Using a standard data engineering pipeline, this project explores interrelationships among variables related to consumer behavior based on a dataset of customer demographics, purchases, and brand performance.

The dataset provides data on consumer purchase activity, social media activity, consumer demographics, and consumers' response to key performance indicators like brand loyalty and customer satisfaction. The project analyzes these data to explore relationships among various key variables, and runs a Random Forest Regressor to identity the variable that is the most strongly correlated to Brand Loyalty. In particular, Brand Loyalty and Customer Satisfaction are variables of special interest to marketing companies, market research firms, and any industry-leading firms. Using a workflow consisting of data cleaning, exploratory data analysis (EDA), predictive modeling, and testing, this project aims to shed light on how various attributes of customers are related to these key variables.

## 2. Dataset

This dataset is sourced from [kaggle.com](https://www.kaggle.com/datasets/salahuddinahmedshuvo/ecommerce-consumer-behavior-analysis-data). It contains detailed consumer behavior information suitable for market research and statistical analysis. It covers purchasing patterns, demographics, product preferences, and outcomes like customer satisfaction and brand loyalty. The data can be applied to tasks such as market segmentation, predictive modeling, and analyzing decision-making processes.

# 3. File & Repository Structure
```
├── .devcontainer/          # Devcontainer configuration files
│ └── devcontainer.json
├── .github/
│ └── workflows/
│ └── CI.yml                # GitHub Actions workflow for CI/CD
├── pycache/                # Compiled Python cache files
├── Makefile                # Commands for building, testing, automation
├── README.md               # Project documentation
├── analysis.py             # Analysis script (refactored!)
├── data.csv                # Dataset (raw input file)
├── requirements.txt        # Dependencies (used for make install)
├── test_analysis.py        # Pytest test cases
├── .pytest_cache/          # pytest cache files
├── test_cases_passed.png   # Screenshot of passing test results
├── titled_chart.png        # Visualization chart
├── screenshots             # screenshots of changes in refactoring
```

# 4. Installation & Setup

### 4.1. Requirements
- python >3.0
- pandas
- numpy
- train_test_split from sklearn.model_selection
- RandomForestRegressor from sklearn.ensemble
- mean_squared_error, r2_score from sklearn.metrics
- seaborn
- matplotlib.pyplot
- black
- flake8

### 4.2. Virtual Environment Instructions: Devcontainer Setup
This project includes a `.devcontainer` configuration for a consistent development environment in VS Code. It can be enabled using the following steps:
1. Open your repository in VS Code.
2. Install the **Dev Containers** extension from the Extensions tab on the left.
3. Reopen the project in the container by clicking on the blue icon in the bottom left corner (choose the option `Remote-Containers: Reopen in Container`).
4. All dependencies will be installed automatically, ensuring reproducibility. To cross-check, make sure that the newly opened environment has a devcontainer in its title.
5. If working locally in VS Code, push all of your changes to GitHub. A new folder called `.devcontainer` would be created in your repository.

# 5. Usage

### How to run the project pipeline:

#### 5.1. make all
This would run the Makefile, which contains commands to run the project workflow. They include the following:

#### 5.2. Installing python dependencies using `make install`
This would install the necessary libraries required to run the project, including formatting and linting libraries.

#### 5.3. make format
This would apply black formatting to all python script files.

#### 5.4. make lint
This would prune unnecessary code and unclean formatting from the files. We will be ignoring trivial errors for extra whitespace and longer lines (Error codes E501,W503).

#### 5.5. make test
This would run the testing file `test_analysis.py`, which tests the file `analysis.py`. Below is a description of the workflow in these files:

### 5.6. `analysis.py`
#### 5.6.1. Data Import & Cleaning
- Loaded the dataset (`data.csv`) into a pandas DataFrame.
- Removed identifiers (`Customer_ID`) and cleaned column names.
- Cleaned numeric columns (e.g., `Purchase_Amount`).
- Encoded categorical variables using one-hot encoding for analysis.

#### 5.6.2. Exploratory Data Analysis (EDA).
- Checked correlations between variables of interest, like Customer Satisfaction and Brand Loyalty.
- Identified Brand Loyalty as a key variable of interest for prediction.
- Defined my research question: "Identify the variable that is the most strongly correlated with Brand Loyalty."

#### 5.6.3. Predictive Modeling
- Framed Brand Loyalty as a categorical target (e.g., low, medium, high).
- Applied Logistic Regression to predict customer loyalty.
- Evaluated model performance with accuracy and classification metrics.

#### 5.6.4. Results
- Among all features, Age showed the strongest correlation with Brand Loyalty.

#### 5.6.5. Visualization
- Used a Box Plot to illustrate the relationship between Age groups and Brand Loyalty.

#### 5.6.6. Data Visualization Plot:
- Used seaborn to generate a boxplot of Age against Brand Loyalty.

#### 5.6.7. Testing
- Testing is implemented with **pytest** to ensure correctness and reliability of the pipeline.  

### 5.7. Test Cases in `test_analysis.py`

#### 5.7.1. test_loading_data
   - Ensures the dataset is loaded correctly into a pandas DataFrame.
   - Verifies the shape of the DataFrame matches expectations (1000 rows × 28 columns).

#### 5.7.2. test_cleaning_data
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

#### 5.7.3. test_exploring_data
   - Confirms that the `explore_data` function returns all expected metrics in a dictionary.
   - Validates the contents of each returned object:
     - Grouped statistics (`loyalty_income`, `loyalty_mediaInfluence`, etc.) are DataFrames with expected structure (contain means, correct indices, non-empty).
     - Correlation outputs (`loyalty_researchTime`, `satis_researchTime`) are floats between -1 and 1.

#### 5.7.4. test_train_random_forest
   - Checks that the Random Forest model is trained correctly:
     - Returned model has a `.predict` method.
     - Performance metrics (MSE, R²) are floats.


#### 5.8. make clean
This deletes compiled Python cache files.


# 6. CICD Pipeline

To maintain consistency and code quality across the pipeline, this project contains a Continuous Integration and Continuous Delivery/Continuous Deployment (CICD) workflow. The GitHub Actions workflow performs CICD in the following way:

- Run Linting using flake8.
- Run formatting using black formatter.
- Run test cases using pytest and testing.py to ensure test coverage after changes to code.


# 7. Results & Visualizations

Age is found to have the largest impact on Brand Loyalty, although the coefficient is fairly low in absolute terms. Below is the visualization chart built by the analysis script.

[titled_chart](titled_chart.png)

# 8. Refactoring & Code Quality

## Improvements during refactoring (renamed variables, helper functions, structured comments, CICD pipeline, among others):
1.	Made helper functions for each comparison in EDA using Extract Method.
[Helper_Function_Screenshot](screenshots/screenshot2.png)
[Code_replaced_screenshot](screenshots/screenshot3.png)

2.	Used Rename to change variable names to var_by_income, loyalty_by_income, etc. Did this in the function definitions, their calls in EDA, as well as variable names in the output.
[Variables_renamed_screenshot](screenshots/screenshot4.png)

3.  Updated the testing file to test newly-added helper functions too, along with previously existing code.
4.	Changed the name of train_random_forest function to run_random_forest, because that function returns the entire output of the ML algorithm, and doesn’t stop at training data.
5.	Removed unnecessarily long comments in analysis.py and improved the overall comment and code structure. Added simple comments to the run_random_forest function because there weren’t any comments in it earlier.
[Code_reorganized_screenshot](screenshots/screenshot1.png)

6.  Added a CICD pipeline and yml file.
7.  Added a status badge to the README file.
8.	Added docstrings to all functions.
9.	Made a function for data visualization.
10.	Updated the README file to remove AI-generated content.

Benefits of Refactoring: better reproducibility and maintainability, among others.

# 9. Future Work

### Possible next steps:
- Try fitting other ML algorithms to get better results.
- Expand the dataset to improve the accuracy of predictions.
- Collect data on sub-indicators of Brand Loyalty, such as Purchase Repetitions and Repeated Visits to Websites. Parse the impact of predictor variables across these sub-indicators by running more analysis.


# 10. Contributors
Ananya Jogalekar (Duke NetID: aj463)
