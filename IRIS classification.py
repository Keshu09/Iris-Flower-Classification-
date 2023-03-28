#!/usr/bin/env python
# coding: utf-8

# ## Import modules
# 

# In[3]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


# ## Loading the dataset
# 

# In[6]:


df = pd.read_csv("C:/Users/aaksh/OneDrive/Desktop/IRIS.csv")
df.head(15)


# In[7]:


df.describe()


# In[8]:


#display information
df.info()


# In[10]:


#to display no. of samples in each class
df['species'].value_counts()


# ## Preprocessing the dataset

# In[11]:


#check null values
df.isnull().sum()


# ## Exploratory Data Analysis

# In[12]:


#histogram
df['sepal_length'].hist()


# In[13]:


df['sepal_width'].hist()


# In[14]:


df['petal_length'].hist()


# In[19]:


df['petal_width'].hist()


# In[27]:


#scatter plot
colors = ['red','pink','blue']
species = ['Iris-setosa','Iris-versicolor','Iris-virginica' ]


# In[30]:


for i in range(3):
    x = df[df['species']==species[i]]
    plt.scatter(x['sepal_length'],x['sepal_width'],c=colors[i],label = species[i])
    plt.xlabel("sepal_length")
    plt.ylabel("sepal_width")
    plt.legend()


# In[31]:


for i in range(3):
    x = df[df['species']==species[i]]
    plt.scatter(x['petal_length'],x['petal_width'],c=colors[i],label = species[i])
    plt.xlabel("petal_length")
    plt.ylabel("petal_width")
    plt.legend()


# In[34]:


for i in range(3):
    x = df[df['species']==species[i]]
    plt.scatter(x['sepal_length'],x['petal_length'],c=colors[i],label = species[i])
    plt.xlabel("sepal_length")
    plt.ylabel("petal_length")
    plt.legend()


# In[35]:


for i in range(3):
    x = df[df['species']==species[i]]
    plt.scatter(x['sepal_width'],x['petal_width'],c=colors[i],label = species[i])
    plt.xlabel("sepal_width")
    plt.ylabel("petal_width")
    plt.legend()


# ## corelation matrix

# In[36]:


df.corr()


# In[39]:


corr = df.corr()
fig, ax =plt.subplots(figsize=(4,4))
sns.heatmap(corr, annot=True , ax=ax)


# ## Label encoder
# 

# In[40]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[43]:


#convert categorical value to numeric value
df['species']=le.fit_transform(df['species'])
df.head(15)


# ## Model Training

# In[67]:


from sklearn.model_selection import train_test_split
#split train and test data
#train = 70
#test = 30
x=df.drop(columns=['species'])
y=df['species']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)


# ## logistic regression

# In[64]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[65]:


model.fit(x_train,y_train)


# In[68]:


#print metric to get performance
print("Accuracy:",model.score(x_test,y_test)*100)  #we get higher accuracy in logistic regression.


# In[70]:


#using decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[71]:


model.fit(x_train,y_train)


# In[72]:


print("Accuracy:",model.score(x_test,y_test)*100)

