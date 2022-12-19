#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression


# In[50]:


price_data = pd.read_csv('price.csv')
price_data = price_data.sample(frac=1) # shuffle


# In[51]:


# ideally should one-hot encode or integer encode but model accuracy is not the focus of this task
price_data.drop(columns=['bed_room'], inplace=True)


# In[52]:


# train test split
from sklearn.model_selection import train_test_split
y=price_data.loc[:,'price']
X=price_data.drop(columns=['price'])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[54]:


from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_test))


# In[59]:


lr=LinearRegression()
lr.fit(imputed_X_train, y_train)


# In[61]:


with open('lr.pickle', 'wb') as model:
    pickle.dump(lr, model)
    print('Object serialised, pickling completed')

