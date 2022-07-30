#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machine Project
# 
# 
# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# 
# **For this series of lectures, we will be using the famous [breast cancer dataset](https://goo.gl/U2Uwz2). 
# The breast cancer dataset is a classic and very easy binary classification dataset. The features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.**
# 
# **Ten real-valued features are computed for each cell nucleus:**
# 
# 1. radius (mean of distances from center to points on the perimeter)
# 2. texture (standard deviation of gray-scale values)
# 3. perimeter
# 4. area
# 5. smoothness (local variation in radius lengths)
# 6. compactness (perimeter^2 / area - 1.0)
# 7. concavity (severity of concave portions of the contour)
# 8. concave points (number of concave portions of the contour)
# 9. symmetry
# 10. fractal dimension

# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


cancer = load_breast_cancer()


# **The data set is presented in a dictionary form:**

# In[4]:


cancer.keys()


# **We can grab information and arrays out of this dictionary to set up our data frame and understanding of the features:**

# In[5]:


print(cancer['DESCR'])


# In[6]:


cancer['feature_names']


# ## Set up DataFrame

# In[7]:


df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.info()


# In[8]:


cancer['target']


# In[14]:


df_feat['target'] = pd.DataFrame(cancer['target'],columns=['Cancer'])


# **Now let's actually check out the dataframe:**

# In[17]:


df_feat.head()


# # Exploratory Data Analysis
# 
# 

# **We'll skip the Data Visualization part for this project since there are so many features that are hard to interpret if you don't have domain knowledge of cancer or tumor cells.**

# ## Train Test Split

# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.30, random_state=101)


# # Train the Support Vector Classifier

# In[20]:


from sklearn.svm import SVC


# In[21]:


model = SVC()


# In[22]:


model.fit(X_train,y_train)


# ## Predictions and Evaluations
# 
# **Now let's predict using the trained model:**

# In[23]:


predictions = model.predict(X_test)


# In[24]:


from sklearn.metrics import classification_report,confusion_matrix


# In[25]:


print(confusion_matrix(y_test,predictions))


# In[26]:


print(classification_report(y_test,predictions))


# **We can improve the results further by using a GridSearch.**

# # Gridsearch
# 
# **Finding the right parameters (like what C or gamma values to use) is a tricky task. But luckily, we can be a little lazy and just try a bunch of combinations and see what works best. This idea of creating a 'grid' of parameters and just trying out all the possible combinations is called a Gridsearch, this method is common enough that Scikit-learn has this functionality built in with GridSearchCV. The CV stands for cross-validation.**
# 
# **GridSearchCV takes a dictionary that describes the parameters that should be tried to train the model. The grid of parameters is defined as a dictionary, where the keys are the parameters and the values are the settings to be tested.**

# In[27]:


param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[28]:


from sklearn.model_selection import GridSearchCV


# **One of the great things about GridSearchCV is that it is a meta-estimator. It takes an estimator like SVC, and creates a new estimator, that behaves exactly the same - in this case, like a classifier. You should add refit=True and choose verbose to whatever number you want. The higher the number, the more verbose (verbose just means the text output describing the process) the output will be.**

# In[29]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)


# **What fit does is a bit more involved then usual. First, it runs the same loop with cross-validation, to find the best parameter combination. Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), to built a single new model using the best parameter setting.**

# In[30]:


grid.fit(X_train,y_train)


# **You can inspect the best parameters found by GridSearchCV in the best_params_ attribute, and the best estimator in the best\_estimator_ attribute:**

# In[31]:


grid.best_params_


# In[32]:


grid.best_estimator_


# **Then you can re-run predictions on this grid object just like you would with a normal model:**

# In[33]:


grid_predictions = grid.predict(X_test)


# In[34]:


print(confusion_matrix(y_test,grid_predictions))


# In[35]:


print(classification_report(y_test,grid_predictions))


# **Voila! The results have slightly improved which was desired.**
