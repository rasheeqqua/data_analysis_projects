#!/usr/bin/env python
# coding: utf-8

# # K Nearest Neighbors Project
# 
# You've been given a classified data set from a company. They've hidden the feature column names but have given you the data and the target classes. 
# 
# We'll try to use KNN to create a model that directly predicts a class for a new data point based off of the features.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data

# In[2]:


df = pd.read_csv('KNN_Project_Data')


# **Check the head of the dataframe.**

# In[3]:


df.head() 


# # Exploratory Data Analysis
# 
# To understand the distribution of these classified data, we'll just do a pairplot with seaborn.
# 
# **We'll use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.**

# In[4]:


sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')


# # Standardize the Variables
# 
# Some of the data might have a huge range (e.g. from 20 to 2,000,000) and some of the data might be closely spaced (e.g. from 10 to 20). To put all of these data in a uniform range we will have to normalize data.

# In[5]:


from sklearn.preprocessing import StandardScaler


# In[6]:


scaler = StandardScaler()


# **Fit scaler to the features:**

# In[7]:


scaler.fit(df.drop('TARGET CLASS',axis=1))


# **We'll use the .transform() method to transform the features to a scaled version:**

# In[8]:


scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# **Now we'll convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

# In[9]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# # Train Test Split

# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30)


# # Using KNN

# In[12]:


from sklearn.neighbors import KNeighborsClassifier


# **Let's create a KNN model instance with n_neighbors=1**

# In[13]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[14]:


knn.fit(X_train,y_train)


# # Predictions and Evaluations

# **Use the predict method to predict values using your KNN model and X_test.**

# In[15]:


pred = knn.predict(X_test)


# In[16]:


from sklearn.metrics import classification_report,confusion_matrix


# In[17]:


print(confusion_matrix(y_test,pred))


# In[18]:


print(classification_report(y_test,pred))


# # Choosing a K Value
# Let's go ahead and use the elbow method to pick a good K Value.
# 
# **We'll have to create a for loop that iterates through various KNN models with different k values, then keep track of the error_rate for each of these models with a list to find the best KNN Model.**

# In[19]:


error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# **Now we'll create the following plot using the information from our loop.**

# In[20]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# ## Retrain with new K Value
# 
# **Let's retrain our model with the best K value and re-do the classification report and the confusion matrix.**

# In[21]:


# NOW WITH K=30
knn = KNeighborsClassifier(n_neighbors=21)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=21')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

