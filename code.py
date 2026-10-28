## Important Libraries--
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

### Loading data---
df = pd.read_csv(r"C:\Users\anand\PrimeBatch\Data\DataSets\HousePricePrediction.csv")

# info about Data--
df.head()
df.shape

## Separating the data---
X = df.drop(columns=["SalePrice", "Id"])
Y = df["SalePrice"]

## EDA portion- 
## Fill the Na with mean value for SalePrice---
Y = Y.fillna(Y.mean())
## Dropeing the null values-- as it is <5%
X= X.dropna()
Y = Y.loc[X.index]  ## this contain those values present in X-- number of data should be same

## Creating the dummy values for all this values --> Encoding
cols = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']
X = pd.get_dummies(X, columns=cols, drop_first= True)

## Extra weightage to the important perameter--
X["TotalBsmtSF"]= X["TotalBsmtSF"]* X["YearBuilt"]* X["OverallCond"]
X["House_Age"] = 2024 - X["YearBuilt"]
# High condition + New age = Premium price
X["Cond_Age_Score"] = X["OverallCond"] / (X["House_Age"] + 1) 
# Proportion of basement that is finished
X["Bsmt_Finish_Ratio"] = X["BsmtFinSF2"] / (X["TotalBsmtSF"] + 1)

X["Log_LotArea"] = np.log1p(X["LotArea"])


## Model train--
#spliting the data-
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

#Train the model on split data-
model = LinearRegression()
model.fit(X_train, y_train)

## Predected value---
y_pred = model.predict(X_test)

## Evalute--
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
r2 = r2_score(y_test, y_pred)
print("R2 Score:",r2)

## Adjasted r^2--
n = X_test.shape[0]
p = X_test.shape[1]
adj_r2 = 1-((1-r2)*(n-1)/(n-p-1))
print("Asjested R2 score:",adj_r2)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print(mean_absolute_percentage_error(y_test, y_pred))


### Feauter Engineering---
pd.set_option('future.no_silent_downcasting', True)
X = X.replace({'True': 1, 'False': 0, True: 1, False: 0})

# Scaling--
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)

## again train the model for better result--
model = LinearRegression()
model.fit(X_train_scl, y_train)
## Predected value---
y_pred = model.predict(X_test_scl)

## Evalute--
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 Score:",r2)

## Adjasted r^2--
n = X_test_scl.shape[0]
p = X_test_scl.shape[1]
adj_r2 = 1-((1-r2)*(n-1)/(n-p-1))
print("Asjested R2 score:",adj_r2)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print(mean_absolute_percentage_error(y_test, y_pred))
