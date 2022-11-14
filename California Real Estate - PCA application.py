#!/usr/bin/env python
# coding: utf-8

# # A Step-by-Step Explanation of PCA on California Estates

# ### Import the libraries

# In[ ]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ### Load the dataset 

# In[ ]:


data = pd.read_csv('California_Real_Estate.csv', sep=';')
df_real_estate = data.copy()
df_real_estate


# ### Discard the rows with NaN values

# In[ ]:


df_real_estate_nonull = df_real_estate[df_real_estate['Status'] == 1]
df_real_estate_nonull


# In[ ]:


scaler = StandardScaler()
df_re_nonull_std = scaler.fit_transform(df_real_estate_nonull)


# In[ ]:


pca = PCA()
pca.fit_transform(df_re_nonull_std)


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


plt.figure(figsize = (11,6))
components = ['Component 1','Component 2','Component 3','Component 4','Component 5','Component 6','Component 7','Component 8']
var_exp = pca.explained_variance_ratio_
plt.bar(components, var_exp)
plt.title('Explained variance by principal components')
plt.xlabel('Principal components')
plt.ylabel('Explained variance ratio')
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,9),pca.explained_variance_ratio_.cumsum(),marker='o', linestyle='--')
plt.title('Explained variance by components')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()


# # PCA Covariance Matrix in Jupyter – Analysis and Interpretation

# In[ ]:


pca = PCA(n_components=4)


# In[ ]:


pca.fit(df_re_nonull_std)


# In[ ]:


pca.components_


# In[ ]:


df_pca_comp = pd.DataFrame(data=pca.components_,
                        columns=df_real_estate.columns.values,
                        index=['Component 1','Component 2','Component 3','Component 4'])
df_pca_comp


# In[ ]:


sns.heatmap(df_pca_comp,
           vmin=-1,
           vmax=1,
           cmap='RdBu',
           annot=True)
plt.yticks([0,1,2,3],
          ['Component 1','Component 2','Component 3','Component 4'],
          rotation=45,
          fontsize=9)

plt.show()


# In[ ]:




