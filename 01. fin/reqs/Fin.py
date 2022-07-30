#!/usr/bin/env python
# coding: utf-8

# # Finance Data Project
# 
# **In this data project we will focus on exploratory data analysis of stock prices. Keep in mind, this project is just meant to practice visualization and pandas skills, it is not meant to be a robust financial analysis or be taken as financial advice.**
# 
# **We'll focus on bank stocks and see how they progressed throughout the [financial crisis](https://en.wikipedia.org/wiki/Financial_crisis_of_2007%E2%80%9308) all the way to early 2016.**

# ## Data
# 
# **In this section we will learn how to use pandas to directly read data from Yoogle Finance using pandas.**
# 
# **The imports:**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set_style('whitegrid')
import plotly
import cufflinks as cf
cf.go_offline()

from pandas_datareader import data, wb
import datetime


# **We will get stock information for the following banks:**
# *  Bank of America
# * CitiGroup
# * Goldman Sachs
# * JPMorgan Chase
# * Morgan Stanley
# * Wells Fargo

# In[2]:


start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 1)


# In[3]:


# Bank of America
BAC = data.DataReader('BAC', 'yahoo', start, end)

# CitiGroup
C = data.DataReader('C', 'yahoo', start, end)

# Goldman Sachs
GS = data.DataReader('GS', 'yahoo', start, end)

# JPMorgan Chase
JPM = data.DataReader('JPM', 'yahoo', start, end)

# Morgan Stanley
MS = data.DataReader('MS', 'yahoo', start, end)

# Wells Fargo
WFC = data.DataReader('WFC', 'yahoo', start, end)


# In[4]:


tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']


# **We'll concatenate the bank dataframes together to a single data frame called bank_stocks. Then we'll set the keys argument equal to the tickers list. Also we need to pay attention to the axis along which the data will be concatenated:**

# In[5]:


bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC],axis=1,keys=tickers)


# **Setting the column name levels:**

# In[6]:


bank_stocks.columns.names = ['Bank Ticker','Stock Info']


# **Checking the head of the bank_stocks dataframe.**

# In[7]:


bank_stocks.head()


# # Exploratory Data Analysis (EDA):
# 
# **Let's explore the data a bit**
# 
# **What is the maximum Close Price for each bank's stock throughout the time period?**

# In[8]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()


# **Let's create a new empty DataFrame called returns. This dataframe will contain the returns for each bank's stock. Returns are typically defined by:**
# 
# $$r_t = \frac{p_t - p_{t-1}}{p_{t-1}} = \frac{p_t}{p_{t-1}} - 1$$

# In[9]:


returns = pd.DataFrame()


# **We can use pandas pct_change() method on the Close Column to create a column representing this return value:**

# In[10]:


for tick in tickers:
    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change()
returns.head()


# **Let's check which stock stands out using pairplot.**

# In[11]:


sns.pairplot(returns[1:])


# Background on [Citigroup's Stock Crash available here.](https://en.wikipedia.org/wiki/Citigroup#November_2008.2C_Collapse_.26_US_Government_Intervention_.28part_of_the_Global_Financial_Crisis.29) 
# 
# **You'll also see the enormous crash in value if you take a look at the stock price plot (which we do later in the visualizations).**

# **Using this returns DataFrame, let's try to figure out on which dates each bank stock had the best and worst single day returns.**

# In[12]:


returns.idxmin()


# **We should notice that Citigroup's largest drop and biggest gain were very close to one another, did anythign significant happen in that time frame?**

# [Citigroup had a stock split.](https://www.google.com/webhp?sourceid=chrome-instant&ion=1&espv=2&ie=UTF-8#q=citigroup+stock+2011+may)

# In[13]:


returns.idxmax()


# **We can see that the biggest gain as well as the biggest loss was experienced during the inaugural ceremony of Barack Obama which is interesting.**

# **After taking a look at the standard deviation of the returns, we can classify as the riskiest investments over the entire time period.**

# In[15]:


returns.std()


# **CitiGroup is the riskiest.**
# 
# **For the year 2015 the riskiest investments would be:**

# In[17]:


returns.loc['2015-01-01':'2015-12-31'].std()


# **Morgan Stanley or Bank of America**

# **2015 returns for Morgan Stanley:**

# In[21]:


sns.displot(returns.loc['2015-01-01':'2015-12-31']['MS Return'],color='green',bins=50)


# **2008 returns for CitiGroup:**

# In[23]:


sns.displot(returns.loc['2008-01-01':'2008-12-31']['C Return'],color='red',bins=50)


# **Close Price for each bank for the entire index of time:**

# In[24]:


for tick in tickers:
    bank_stocks[tick]['Close'].plot(figsize=(12,4),label=tick)
plt.legend()


# In[25]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').plot()


# In[26]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').iplot()


# ## Moving Averages
# 
# **Let's analyze the moving averages for these stocks in the year 2008.**
# 
# **Rolling 30 day average against the Close Price for Bank of America's stock for the year 2008:**

# In[27]:


plt.figure(figsize=(12,6))
BAC['Close'].loc['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
BAC['Close'].loc['2008-01-01':'2009-01-01'].plot(label='BAC CLOSE')
plt.legend()


# **Heatmap of the correlation between the stocks Close Price:**

# In[28]:


sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)


# **Let's use seaborn's clustermap to cluster the correlations together:**

# In[29]:


sns.clustermap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)


# In[30]:


close_corr = bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr()
close_corr.iplot(kind='heatmap',colorscale='rdylbu')


# **Now we will rely on the cufflinks library to create some Technical Analysis plots.**

# **Candle plot of Bank of America's stock from Jan 1st 2015 to Jan 1st 2016:**

# In[32]:


BAC[['Open', 'High', 'Low', 'Close']].loc['2015-01-01':'2016-01-01'].iplot(kind='candle')


# **Simple Moving Averages plot of Morgan Stanley for the year 2015:**

# In[34]:


MS['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages')


# **Bollinger Band Plot for Bank of America for the year 2015:**

# In[35]:


BAC['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='boll')


# **Definitely a lot of more specific finance topics here, so don't worry if you didn't understand them all! The only thing you should be concerned with understanding are the basic pandas and visualization oeprations.**
