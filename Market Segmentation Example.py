#!/usr/bin/env python
# coding: utf-8

# ## Market Segmentation Example

# ### Import the Relevant Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()    # set style to seaborn

from sklearn.cluster import KMeans


# ### Load the Data

# In[2]:


data = pd.read_csv('3.12. Example.csv')


# In[3]:


data #print data


# In[4]:


# Satisfaction: self-reported on a scale of 1 through 10 with 10 being Extremely Satisfied
# Type of Data: DISCRETE
# Range: 1 to 10

# Brand Loyalty: no widely accpeted technique to measure it but there are proxies like churn rate, retention rate, or customer lifetime value (CLV)
# Type of Data: CONTINUOUS
# Range: -2.5 to 2.5


# ### Plot the Data

# In[5]:


plt.scatter(data['Satisfaction'],data['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')


# ### Select the Features

# In[6]:


# Can divide the plot into 4 equal squares:
# 1. Low Satisfaction, Low Loyalty
# 2. Low Satisfaction, High Loyalty
# 3. High Satisfaction, Low Loyalty
# 4. High Satisfaction, High Loyalty


# In[7]:


x = data.copy()


# ### Clustering

# In[8]:


kmeans = KMeans(2)
kmeans.fit(x)


# ### Clustering Results

# In[9]:


clusters = x.copy()
clusters['cluster_pred'] = kmeans.fit_predict(x)


# In[10]:


plt.scatter(clusters['Satisfaction'],clusters['Loyalty'],c=clusters['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')


# In[11]:


# most probably, the algorithm ONLY considered 'Satisfaction' as a feature
# this is due to not standardizing before clustering
# whenever we cluster on the basis of a single fature, the results looks like this graph; as if a vertical line was drawn to cluster


# ### Standardize the Variables

# In[12]:


# to give the features equal weight, we must standardize the data before clustering

from sklearn import preprocessing
# 'sklearn.preprocessing.scale(x)' scales (standardizes with mean 0, and st. dev. of 1 by default) each variable (column) separately
x_scaled = preprocessing.scale(x)
x_scaled


# ### Take Advantage of the Elbow Method

# In[13]:


# because we do not know the optimal number of clusters to use, we will invoke the Elbow Method to determine
wcss =[] #declare a list

for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
    
wcss


# In[14]:


plt.plot(range(1,10),wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')


# In[15]:


# from the plot, we can infer 4 different clusters to try (2,3,4,5)
# we still don't know which is the best on to use


# ### Explore Clustering Solutions and Select the Number of Clusters

# In[16]:


kmeans_new = KMeans(4)  #let's try a cluster of '2' first
kmeans_new.fit(x_scaled) #fit the data
clusters_new = x.copy() #create a new dataframe
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled) #create new column with this new variable


# In[17]:


clusters_new


# In[18]:


# we will plot the data without standardizing the axes, but the solution will be the standardized one

plt.scatter(clusters_new['Satisfaction'],clusters_new['Loyalty'], c=clusters_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
