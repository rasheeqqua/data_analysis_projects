#!/usr/bin/env python
# coding: utf-8

# ## Database Description
# 
# The dataset consists of house prices from King County an area in the US State of Washington. The dataset was obtained from [Kaggle.](https://www.kaggle.com/datasets/swathiachath/kc-housesales-data)
# The dataset consists of 21 variables and 21613 observations.
# 
# * id - a notation for a house
# * date - Date house was sold
# * price - Price is prediction target
# * bedrooms - Number of Bedrooms/House
# * bathrooms - Number of bathrooms/bedrooms
# * sqftliving - square footage of the home Numeric sqftlot square footage of the lot
# * floors - Total floors (levels) in house
# * waterfront - House which has a view to a waterfront
# * view - Has been viewed
# * condition - How good the condition is ( Overall ). 1 indicates worn out property and 5 excellent.(http://info.kingcounty.gov/assessor/esales/Glossary.aspx?type=r#g) 
# * grade - overall grade given to the housing unit, based on King County grading system. 1 poor ,13 excellent
# * sqftabove - square footage of house apart from basement Numeric sqftbasement square footage of the basement
# * yrbuilt - Built Year Numeric yrrenovated Year when house was renovated
# * zipcode - zip
# * lat - Latitude coordinate
# * long - Longitude coordinate
# * sqftliving15 - Living room area in 2015(implies-- some renovations). This might or might not have affected the lotsize area.
# * sqftlot15 - lotSize area in 2015(implies-- some renovations).
# 
# **Predict the sales of houses in King County with an accuracy of at least 75-80% and       understand which factors are responsible for higher property value - $650K and above.**

# ### Basic Imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('kc_house_data.csv')


# #### We'll remove all the rows that have null values

# In[3]:


df = df.dropna()


# ### Exploratory Data Analysis

# **We'll try to see the distribution of number of houses based on their price**

# In[6]:


plt.figure(figsize=(14,8))
sns.histplot(df['price'], bins=50, kde=True)


# **Looks like most of the houses are in the 0.5 - 1 million dollar price range**

# In[5]:


plt.figure(figsize=(14,8))
sns.countplot(x = df['bedrooms'])


# In[7]:


df.corr()['price'].sort_values()


# In[8]:


plt.figure(figsize=(14,8))
sns.scatterplot(x = 'price', y = 'sqft_living', data = df)


# In[9]:


plt.figure(figsize=(14,8))
sns.boxplot(x = 'bedrooms', y = 'price', data = df)


# In[9]:


plt.figure(figsize=(14,8))
sns.scatterplot(x = 'price', y = 'long', data = df)


# In[10]:


plt.figure(figsize=(14,8))
sns.scatterplot(x = 'price', y = 'lat', data = df)


# In[11]:


plt.figure(figsize=(14,8))
sns.scatterplot(x = 'long', y = 'lat', data = df, hue = 'price', palette = 'viridis')


# In[12]:


non_top_1_percentage = df.sort_values('price', ascending = False).iloc[215:]
non_top_1_percentage.head()


# In[13]:


plt.figure(figsize=(14,8))
sns.scatterplot(x = 'long', y = 'lat', data = non_top_1_percentage, 
                           edgecolor = None, alpha = 1, hue = 'price', palette = 'RdYlGn')


# In[14]:


plt.figure(figsize=(14,8))
sns.boxplot(x = 'waterfront', y = 'price', data = df)


# ### Feature Engineering Process

# In[15]:


df.head()


# In[16]:


df = df.drop('id', axis = 1)


# In[17]:


df.head()


# In[18]:


df['date']


# In[19]:


df['date'] = pd.to_datetime(df['date'])


# In[20]:


df['date']


# In[21]:


df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)


# In[22]:


df.head()


# In[23]:


plt.figure(figsize=(14,8))
sns.boxplot(x = 'month', y = 'price', data = df)


# In[24]:


plt.figure(figsize=(14,8))
df.groupby('month').mean()['price'].plot()


# In[25]:


plt.figure(figsize=(14,8))
df.groupby('year').mean()['price'].plot()


# In[26]:


df = df.drop('date', axis = 1)


# In[27]:


df.head()


# In[28]:


df['zipcode'].value_counts()


# In[29]:


df = df.drop('zipcode', axis = 1)


# In[30]:


df['yr_renovated'].value_counts()


# In[31]:


df['sqft_basement'].value_counts()


# ### Data Preprocessing and Training

# In[32]:


X = df.drop('price', axis = 1).values
y = df['price'].values


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[35]:


from sklearn.preprocessing import MinMaxScaler


# In[36]:


scaler = MinMaxScaler()


# In[37]:


X_train = scaler.fit_transform(X_train)


# In[38]:


X_test = scaler.transform(X_test)


# In[39]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# In[40]:


X_train.shape


# In[41]:


model = Sequential()

model.add(Dense(19, activation = 'relu'))
model.add(Dense(19, activation = 'relu'))
model.add(Dense(19, activation = 'relu'))
model.add(Dense(19, activation = 'relu'))

model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mse')


# In[42]:


model.fit(x = X_train, y = y_train,
          validation_data = (X_test, y_test),
          batch_size = 128, epochs = 400)


# ### Model Evaluation and Prediction

# In[46]:


losses = pd.DataFrame(model.history.history)


# In[50]:


plt.figure(figsize=(14,8))
losses.plot()


# In[52]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score


# In[53]:


predictions = model.predict(X_test)


# In[56]:


np.sqrt(mean_squared_error(y_test, predictions))


# In[57]:


mean_absolute_error(y_test, predictions)


# In[59]:


explained_variance_score(y_test, predictions)


# In[60]:


plt.figure(figsize=(14,8))
plt.scatter(y_test, predictions)
plt.plot(y_test, y_test, 'r')


# In[63]:


single_house = df.drop('price', axis = 1).iloc[0]
single_house


# In[65]:


single_house = scaler.transform(single_house.values.reshape(-1,19))


# In[67]:


model.predict(single_house)


# In[68]:


df.head(1)


# In[ ]:




