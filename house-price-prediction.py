#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('train.csv', delimiter=',')
df.drop('Id', axis = 1, inplace = True) 


# In[3]:


test = pd.read_csv('test.csv', delimiter=',')
test.drop('Id', axis = 1, inplace = True)


# In[4]:


#Delete columns having NaN more than 30%
for cols in df.columns:
    if df[cols].isnull().sum() >= len(df)*0.3:
        df.drop(cols, axis = 1, inplace = True)


# In[5]:


#Choose the most related numeric columns
data = df.corr()
data = data.iloc[-1,:]
data.drop('SalePrice', inplace = True)  
for n in range(len(data)):
    data.iloc[n] = abs(data.iloc[n])
categories = data.nlargest(12).index
categories = categories.drop('GarageArea')
categories = categories.drop('1stFlrSF')
print(categories)


# In[ ]:


#Identify the categorical columns
cats = []
for n,cols in enumerate(df.columns):
    for i in range(len(df)):
        if df[cols][i]:
            if df[cols][i] == str(df[cols][i]):
                cats.append(cols)
                break
        else:
            i+=1 


# In[ ]:


#Choose the related categorical columns and create the final columns list
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
from statsmodels.formula.api import ols
choice = []
for cat in cats:
    mod = ols('SalePrice ~ {}'.format(cat), data=df).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    choice.append([cat,aov_table['PR(>F)'][0]])  
def Sort(sub_li): 
    sub_li.sort(key = lambda x: x[1]) 
    return sub_li 
choice = Sort(choice)

choice = [choice[x][0] for x in range(0,5)]
print(choice)
#Merge the columns
final_columns = np.array(choice+pd.Series.tolist(categories))


# In[ ]:


#For some visualization
"""
for i in pd.Series.tolist(categories):
    data2 = pd.concat([df['SalePrice'], df[i]], axis=1)
    data2.plot.scatter(x=i, y='SalePrice', ylim=(0,800000));
"""


# In[7]:


#Removing the outliers
df.drop(df[df['GrLivArea'] > 4000].index, inplace = True)
df.drop(df[df['TotalBsmtSF'] > 4000].index, inplace = True)
df.drop(df[df['MasVnrArea'] > 1200].index, inplace = True)


# In[8]:


# Filling the train data
df2 = pd.concat([df,test], axis = 0)
df2 = df2[final_columns].copy()
X = df2.iloc[:,:].values
Y = df.iloc[:, -1].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X[:, 0:5])
X[:, 0:5] = imputer.transform(X[:, 0:5])

imputer2 = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer2.fit(X[:, 5:15])
X[:, 5:15] = imputer2.transform(X[:, 5:15])


# In[9]:


#Encoding the categorical data (encodes both test and train data)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,2,3,4])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[10]:


#Spliting back the encoded data into train and test
test_X = X[len(df):,:]
X = X[:len(df),:]


# In[11]:


#Training the model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, Y)


# In[12]:


#Getting the preditions
y_pred = regressor.predict(test_X)
np.set_printoptions(precision=2)


# In[14]:


#Saving to the csv file
percentile_list = pd.DataFrame(y_pred,np.array([x+1461 for x in range(1459)]))
percentile_list.to_csv('results.csv')
print(y_pred)





