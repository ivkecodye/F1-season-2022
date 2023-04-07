#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries and Dataset

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
dataset = pd.read_csv('Monaco GP.csv')
dataset.head(20)


# In[2]:


# X = Drivers ID, y = Final Position, Z = Constructors ID, H = Grid

X = dataset.iloc[:-3,2:-6].values
y = dataset.iloc[:-3,:-8].values 
Z = dataset.iloc[:-3,4:-4].values
H = dataset.iloc[:-3,8:].values


# In[3]:


# Machine Learning Regression Grid VS Final Position 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(H, y)


# In[4]:


# Monaco GP Driver Position

plt.figure(figsize=(15,10))
plt.grid(color='green', linestyle='--', linewidth='0.5')
plt.scatter(X, y, color = 'blue')
plt.title('Monaco GP Driver Position')
plt.xlabel('Driver ID')
plt.ylabel('Position')


# In[5]:


# Monaco GP Constructor Position

plt.figure(figsize=(15,10))
plt.grid(color='green', linestyle='--', linewidth='0.5')
plt.scatter(Z, y, color = 'blue')
plt.title('Monaco GP Constructor Position')
plt.xlabel('Constructor ID')
plt.ylabel('Position')


# In[6]:


# Monaco GP Driver Points

f, ax = plt.subplots(figsize=(15, 10))
sns.barplot(x='points', y='driver', data=dataset)
plt.title('Monaco GP Driver Points')
plt.xlabel('GP Points')
plt.ylabel('Drivers')


# In[7]:


# Monaco GP Constructor Points

f, ax = plt.subplots(figsize=(15, 10))
sns.barplot(x='points', y='constructor', data=dataset)
plt.title('Monaco GP Constructor Points')
plt.xlabel('GP Points')
plt.ylabel('Constructors')


# In[8]:


# Monaco GP Grid VS Final Position

plt.figure(figsize=(15, 10))
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.scatter(H, y, color = 'red')
plt.plot(H, regressor.predict(H), color = 'blue')
plt.title('Grid VS Finish')
plt.xlabel('Start Position')
plt.ylabel('Finish')

