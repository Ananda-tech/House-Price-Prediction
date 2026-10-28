
# House Price Prediction Using Machine Learning

## Project Overview

This project focuses on predicting house sale prices using Machine Learning and Linear Regression.

The model uses housing-related features such as lot area, year built, overall condition, basement area, and other property characteristics to estimate the sale price of a house.

The project covers data preprocessing, missing-value handling, categorical encoding, feature engineering, model training, and performance evaluation. It also compares model evaluation results before and after applying feature scaling.

## Objectives

- Analyze housing data and prepare it for machine learning.
- Handle missing values in the target variable and input features.
- Convert categorical variables into numerical representations.
- Create new features to capture additional relationships between housing characteristics.
- Train a Linear Regression model to predict house prices.
- Evaluate predictions using multiple regression metrics.
- Apply feature scaling and compare model performance.

## Technologies and Libraries

- **Python** — Programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical data visualization
- **Scikit-learn** — Model training, data splitting, feature scaling, and evaluation

## Project Workflow

### 1. Data Loading and Inspection

The housing dataset is loaded using Pandas.

Initial inspection is performed to examine the dataset's structure, dimensions, and sample records.

### 2. Separating Features and Target

The dataset is divided into input features and the target variable:

- **Input features (X):** Housing characteristics used for prediction, excluding `SalePrice` and `Id`.
- **Target variable (Y):** `SalePrice`, representing the house's sale price.

### 3. Missing-Value Handling

The project handles missing values using the following steps:

- Missing values in the target variable are filled using the code's specified mean-based approach.
- Rows containing missing values in the input features are removed.
- The target variable is aligned with the remaining input rows to maintain matching records.

### 4. Categorical Encoding

Categorical columns are converted into numerical representations using Pandas `get_dummies()`.

The following columns are encoded:

- `MSZoning`
- `LotConfig`
- `BldgType`
- `Exterior1st`

The `drop_first=True` parameter is used to avoid retaining every dummy column for each encoded feature.

### 5. Feature Engineering

Additional features are created to represent housing characteristics in different ways.

| Feature | Description |
|---|---|
| `TotalBsmtSF` | Modified by multiplying its original value by `YearBuilt` and `OverallCond` |
| `House_Age` | Calculated as 2024 minus `YearBuilt` |
| `Cond_Age_Score` | Represents `OverallCond` divided by `House_Age + 1` |
| `Bsmt_Finish_Ratio` | Calculated as `BsmtFinSF2` divided by the modified `TotalBsmtSF + 1` |
| `Log_LotArea` | Log-transformed `LotArea` using NumPy's `log1p()` |

These engineered features are intended to represent additional relationships that may be useful for house price prediction.

### 6. Train-Test Split

The dataset is divided into training and testing sets using Scikit-learn's `train_test_split()`.

- Training data: 80%
- Testing data: 20%
- Random state: 42

The training set is used to fit the model, while the testing set is used to evaluate predictions on unseen records.

### 7. Model Training

**Linear Regression** is used as the primary machine learning algorithm.

The model learns relationships between the input features and house sale prices to generate predictions for the test dataset.

### 8. Model Evaluation

The model is evaluated using the following regression metrics:

- **R² Score:** Measures how much of the variation in sale prices is explained by the model.
- **Adjusted R² Score:** Adjusts R² according to the number of input features and observations.
- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted prices.
- **Root Mean Squared Error (RMSE):** Measures prediction error while penalizing larger errors more heavily.
- **Mean Absolute Percentage Error (MAPE):** Measures prediction error as a proportion of actual prices.

These metrics provide different perspectives on the model's predictive performance.

### 9. Feature Scaling and Model Re-evaluation

The project uses Scikit-learn's `StandardScaler` to standardize the input features.

The scaler is fitted on the training data and applied to both the training and testing sets.

A second Linear Regression model is trained using the scaled features. The same evaluation metrics are then calculated to compare its results with the initial model.

## Installation and Setup

### Prerequisites

- Python 3.9 or later
- Jupyter Notebook or JupyterLab
- Basic knowledge of Python and machine learning

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
```

Navigate to your project directory:

```bash
cd <your-repository-folder>
```

### Step 2: Install the Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Step 3: Add the Dataset

Place `HousePricePrediction.csv` inside your project directory or an appropriate data folder.

Update the dataset path in your code to match the location of the CSV file.

For example:

```python
df = pd.read_csv("data/HousePricePrediction.csv")
```

### Step 4: Run the Project

Open the notebook in Jupyter Notebook or JupyterLab and execute the cells in order.

If your code is saved as a Python script, run it using:

```bash
python house_price_prediction.py
```

Replace the filename with your actual Python script name.

## Evaluation Metrics Summary

The project evaluates the model using the following metrics:

| Metric | Purpose |
|---|---|
| R² Score | Evaluates the proportion of target variance explained by the model |
| Adjusted R² Score | Evaluates model fit while accounting for the number of features |
| MAE | Measures average absolute prediction error |
| RMSE | Measures prediction error with greater sensitivity to large errors |
| MAPE | Measures average absolute percentage prediction error |

Actual evaluation results can be added to this section after running the code.

## Key Learnings

Through this project, I gained practical experience with:

- Preparing a dataset for regression analysis.
- Handling missing values and aligning features with target values.
- Encoding categorical variables.
- Creating engineered features from existing columns.
- Training a Linear Regression model.
- Splitting data into training and testing sets.
- Applying feature scaling with StandardScaler.
- Evaluating regression models using multiple performance metrics.
- Comparing model results before and after feature scaling.

## Limitations and Future Improvements

Potential improvements include:

- Correcting the missing-value imputation logic and validating the target variable before training.
- Performing exploratory data analysis using distributions, scatter plots, and correlation heatmaps.
- Using a preprocessing pipeline to make the workflow more reproducible.
- Fitting preprocessing steps using training data only to avoid data leakage.
- Comparing Linear Regression with Ridge, Lasso, Random Forest, and other regression algorithms.
- Using cross-validation and hyperparameter tuning where appropriate.
- Comparing feature-engineering strategies using a consistent test set.
- Saving the trained model and preprocessing components for future predictions.
- Adding actual evaluation results and visualizations to the repository.

## Disclaimer

This project is developed for educational purposes to demonstrate the application of machine learning to house price prediction.

The predictions depend on the dataset and the features used to train the model. They should not be treated as professional property valuations.

## Author

**Ananda**

Aspiring Data Scientist | Python | Data Analysis | Machine Learning
