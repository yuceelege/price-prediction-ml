# price-prediction-ml
This is a program that accurately predicts the house prices by using data provided by Kaggle

## Prerequisite Libraries
-Numpy <br/>
-Pandas <br/>
-Sci-kit Learn <br/>
-Statsmodels

**(Optional)** <br/>
-Matplotlib <br/>
-Seaborn

## Method

**Data Cleaning** 

- Columns having more than 15% null is erased. <br/>
- Null values in categorical columns are replaced with the most frequent term. <br/>
- Null values in numerical columns are replaced with the mean of the entire column.

**Column Selection**

-Correlation matrix with target parameter is created for numerical columns and top 10 related columns are chosen.
-P values of the categorical columns raleting with SalePrice is identified by using statsmodels and <br/>
and most related 5 five columns are chosen.

**Regression Model** <br/>
Random Forest Regression is used.

**Accuracy**
Relative Squared Error : 0.15962 <br/>
Code can be submitted to the given Kaggle link to get the results: https://www.kaggle.com/c/house-prices-advanced-regression-techniques


