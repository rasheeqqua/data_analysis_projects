#!/usr/bin/env python
# coding: utf-8

# # Movie Recommender Systems Project
# 
# **In this project we will develop basic recommendation systems using Python and pandas.**
# 
# **In this notebook, we will focus on providing a basic recommendation system by suggesting items that are most similar to a particular item, in this case, movies. Keep in mind, this is not a true robust recommendation system. To describe it more accurately,it just tells you what movies/items are most similar to your movie choice.**
# 
# 
# ## Import Libraries

# In[1]:


import numpy as np
import pandas as pd


# ## Get the Data

# In[2]:


column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)


# In[3]:


df.head()


# **Now let's get the movie titles:**

# In[4]:


movie_titles = pd.read_csv("Movie_Id_Titles")
movie_titles.head()


# **We can merge them together:**

# In[5]:


df = pd.merge(df,movie_titles,on='item_id')
df.head()


# # Exploratory Data Analysis
# 
# **Let's explore the data a bit and get a look at some of the best rated movies.**
# 
# 
# ## Visualization Imports

# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')


# **Let's create a ratings dataframe with average rating and number of ratings:**

# In[7]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[8]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[9]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()


# **Now set the number of ratings column:**

# In[10]:


ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()


# **Now a few histograms:**

# In[11]:


plt.figure(figsize=(14,5))
ratings['num of ratings'].hist(bins=50)


# In[12]:


plt.figure(figsize=(14,5))
ratings['rating'].hist(bins=50)


# In[13]:


sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)


# **Now that we have a general idea of what the data looks like, let's move on to creating a simple recommendation system:**

# ## Recommending Similar Movies

# **Now let's create a matrix that has the user ids on one axis and the movie title on another axis. Each cell will then consist of the rating the user gave to that movie.**
# 
# *Note there will be a lot of NaN values, because most people have not seen most of the movies.*

# In[14]:


moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()


# **Most rated movie:**

# In[15]:


ratings.sort_values('num of ratings',ascending=False).head(10)


# **Let's choose two movies: Star Wars (a sci-fi movie) and Liar Liar (a comedy movie).**

# In[16]:


ratings.head()


# **Now let's grab the user ratings for those two movies:**

# In[17]:


starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()


# **We can then use corrwith() method to get correlations between two pandas series:**

# In[18]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)


# **Let's clean this by removing NaN values and using a DataFrame instead of a series:**

# In[19]:


corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()


# **Now if we sort the dataframe by correlation, we should get the most similar movies, however note that we get some results that don't really make sense. This is because there are a lot of movies only watched once by users who also watched star wars (it was the most popular movie).**

# In[20]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# **Let's fix this by filtering out movies that have less than 100 reviews (this value was chosen based off the histogram from earlier).**

# In[21]:


corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()


# **Now sort the values and notice how the titles make a lot more sense:**

# In[22]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()


# **Now the same for the comedy Liar Liar:**

# In[23]:


corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()

