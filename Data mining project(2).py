#!/usr/bin/env python
# coding: utf-8

# # Jones - Data analysis exercise
# 
# **Author:**
# <br>Adi Matalon, 28.10.18

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use("ggplot")
from scipy import interp

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load Data

# In[4]:


pd.set_option('display.max_columns', 100)


# In[5]:


df = pd.read_csv("results.csv")
df


# # Step 1 - Data Exploration

# In[6]:


print('Data Label Distribution:\n')
classes_dist = pd.concat([pd.DataFrame(df['win?'].value_counts()), pd.DataFrame(df['win?'].value_counts())["win?"] / df.shape[0]], axis = 1)
classes_dist.columns = ["NumOfRows" , "RowsPercent"]
print (classes_dist)

df['win?'].hist(bins=np.arange(3) - 0.5, width=1/1.5)
plt.xticks([0,1], ['lost', 'win'])
plt.xlim([-1, 2])
plt.title('Data Targets Distribution')
plt.xlabel('Target Class')
plt.ylabel('Number of Records')
plt.show()


# We can declare that the classes are well-balanced.

# ## Explore the Features
# ### Description and Box Plot

# In[7]:


df = df.drop(['FG Made-Attempted 1', 'FG Made-Attempted 2', '3PT Made-Attempted 1', '3PT Made-Attempted 2',
              'FT Made-Attempted 1', 'FT Made-Attempted 2'], axis=1)


# In[8]:


df.head(n=2)


# In[9]:


df.info()


# All features are non-null, int64 dtype.

# In[10]:


df.describe()  # Get the mathematical description of the data


# In[11]:


df = df.drop(['Flagrant Fouls 1', 'game'], axis=1)


# In[12]:


df.shape[1]


# In[13]:


features = df.drop(['win?'], axis=1).columns


# In[14]:


counter=0
for i in range(6, df.shape[1]+2, 5):
    plt.figure(figsize=(40, 20))
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=20)
    df[features].iloc[:,counter:i].boxplot()
    counter=i


# ### Distribution of each feature within the classes

# In[15]:


for col in features[1:]:
    plt.figure(figsize=(10,3))
    sns.kdeplot(df.loc[df['win?']==0, col], label='loss')  
    sns.kdeplot(df.loc[df['win?']==1, col], label='win')  
    plt.legend(fontsize=22)
    plt.title(col, fontsize=22);


# we can see that in some features the distributions of loss and win are similar. <br>
# it means that it will be difficult to separate between the classes and maybe 100% accuracy is not even possible.

# ### Understand the features correlation

# In[16]:


fig, ax = plt.subplots(figsize=(15,10))
cor_mat = df.corr()
sns.heatmap(cor_mat)


# We cannot see any significant correlation.

# ----------

# ----------------

# # Step 3 - Feature Selection

# ### Desicion Tree

# In[18]:


from sklearn.tree import DecisionTreeClassifier

# Create a Decision tree classifier
tree = DecisionTreeClassifier(criterion= 'gini', splitter='best', max_depth=24, min_samples_split=2)

# Train the classifier
tree = tree.fit(df[features], df[['win?']])

# Get the feature's importance
importances = tree.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for i, feature in enumerate(df[features].columns[indices]):
    print("{}. {} ({:.2f}%)".format(i+1, feature, importances[indices][i]*100))
    
# Plot the feature importances
plt.figure(figsize=(10,7))
plt.title('Feature importance - Desecion Tree')
#plt.title('Feature importances')
plt.bar(range(df[features].shape[1]), importances[indices],
       color="r", align="center")
plt.xticks(range(df[features].shape[1]), df[features].columns[indices], rotation='vertical')
plt.xlim([-1, df[features].shape[1]])
plt.show()


# In[ ]:


we can see that 70% imapcted by assits 2 and 30% by Defensive Rebound 1 on the win of Warriors.

