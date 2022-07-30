#!/usr/bin/env python
# coding: utf-8

# # Decision Trees and Random Forest Project
# 
# **You can check out this [blog post](https://medium.com/@josemarcialportilla/enchanted-random-forest-b08d418cb411#.hh7n1co54) to get a better understaning of decision trees and random forests.**
# 
# For this project we will be exploring publicly available data from [LendingClub.com](www.lendingclub.com). Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help in predicting this.
# 
# Lending club had a [very interesting year in 2016](https://en.wikipedia.org/wiki/Lending_Club#2016), so let's check out some of their data and keep the context in mind. This data is from before they even went public.
# 
# We will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full. It's recommended you use the csv provided as it has been cleaned of NA values.
# 
# Here are what the columns represent:
# * credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# * purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# * int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# * installment: The monthly installments owed by the borrower if the loan is funded.
# * log.annual.inc: The natural log of the self-reported annual income of the borrower.
# * dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# * fico: The FICO credit score of the borrower.
# * days.with.cr.line: The number of days the borrower has had a credit line.
# * revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# * revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# * inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# * delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# * pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# 
# **We'll read the data from "loan_data.csv" file:**

# In[2]:


loans = pd.read_csv('loan_data.csv')


# **Let's check out the details of this dataset:**

# In[3]:


loans.info()


# In[4]:


loans.describe()


# In[5]:


loans.head()


# # Exploratory Data Analysis
# 
# **We'll create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome:**

# In[6]:


plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# **Now we'll create a similar figure, except this time the data will be selected by the not.fully.paid column:**

# In[7]:


plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# **Now we'll show the counts of loans by purpose, with the color hue defined by not.fully.paid:**

# In[8]:


plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')


# **Let's see the trend between FICO score and interest rate:**

# In[9]:


sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')


# **We'll now try to see if the trend differed between not.fully.paid and credit.policy:**

# In[10]:


plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')


# # Setting up the Data
# 
# Let's get ready to set up our data for our Random Forest Classification Model:

# In[11]:


loans.info()


# ## Categorical Features
# 
# Notice that the **purpose** column is in categorical form.
# 
# That means we need to transform them using dummy variables so sklearn will be able to understand them. Let's do this in one clean step using pd.get_dummies.
# 
# Let's show you a way of dealing with these columns that can be expanded to multiple categorical features if necessary:

# In[12]:


cat_feats = ['purpose']


# In[13]:


final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)


# In[14]:


final_data.head()


# ## Train Test Split
# 
# Now its time to split our data into a training set and a testing set:

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# ## Training a Decision Tree Model
# 
# Let's start by training a single decision tree first:

# In[17]:


from sklearn.tree import DecisionTreeClassifier


# In[18]:


dtree = DecisionTreeClassifier()


# In[19]:


dtree.fit(X_train,y_train)


# ## Predictions and Evaluation of Decision Tree

# In[20]:


predictions = dtree.predict(X_test)


# In[21]:


from sklearn.metrics import classification_report,confusion_matrix


# In[22]:


print(classification_report(y_test,predictions))


# In[23]:


print(confusion_matrix(y_test,predictions))


# ## Tree Visualization
# 
# Scikit learn actually has some built-in visualization capabilities for decision trees, we won't use this often and it requires the installation the pydot library, but here is an example of what it looks like and the code to execute this:

# In[25]:


from IPython.display import Image  
from six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(final_data.columns[1:])
features


# In[26]:


dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data, feature_names=features, filled=True, rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png()) 


# ## Training the Random Forest model
# 
# Now its time to train our model:

# In[27]:


from sklearn.ensemble import RandomForestClassifier


# In[28]:


rfc = RandomForestClassifier(n_estimators=600)


# In[29]:


rfc.fit(X_train,y_train)


# ## Predictions and Evaluation

# In[30]:


predictions = rfc.predict(X_test)


# In[31]:


from sklearn.metrics import classification_report,confusion_matrix


# In[32]:


print(classification_report(y_test,predictions))


# In[33]:


print(confusion_matrix(y_test,predictions))


# **What performed better the random forest or the decision tree?**

# Depends what metric we are trying to optimize.
# Notice the recall for each class for the models.
# Neither did very well, more feature engineering is needed.
