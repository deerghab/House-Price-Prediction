#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn 
import seaborn
import statsmodels.formula.api as sm
import numpy as np


# In[5]:


os.chdir("C:\\Users\\91721\\Desktop\\practice")


# In[7]:


house=pd.read_csv("HousingData.csv")


# In[8]:


house.head()


# In[9]:


house.dtypes


# In[10]:


house1=pd.get_dummies(house,drop_first=True)


# In[11]:


house1.head()


# In[12]:


house1.dtypes


# In[13]:


x=house1['SellingPrice000s']
print(x)


# In[14]:


y=house1.drop("SellingPrice000s", axis=1)
print(y)


# In[15]:


seaborn.pairplot(house1,kind='reg')


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.20,random_state=5)


# In[22]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x, y)


# In[ ]:


lm.summary()


# In[ ]:




