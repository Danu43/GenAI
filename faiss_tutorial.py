#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Install Packages
get_ipython().system('pip install faiss-cpu')
get_ipython().system('pip install sentence-transformers')


# In[2]:


# import necessary libraries
import pandas as pd
pd.set_option('display.max_colwidth', 100)


# In[3]:


df = pd.read_csv("sample_text.csv")
df.shape


# In[49]:


df


# ### Step 1 : Create source embeddings for the text column

# In[5]:


from sentence_transformers import SentenceTransformer


# In[6]:


encoder = SentenceTransformer("all-mpnet-base-v2")
vectors = encoder.encode(df.text)


# In[7]:


vectors.shape


# In[8]:


dim = vectors.shape[1]
dim


# ### Step 2 : Build a FAISS Index for vectors

# In[9]:


import faiss

index = faiss.IndexFlatL2(dim)


# ### Step 3 : Normalize the source vectors (as we are using L2 distance to measure similarity) and add to the index

# In[10]:


index.add(vectors)


# In[11]:


index


# ### Step 4 : Encode search text using same encorder and normalize the output vector

# In[64]:


search_query = "I want to buy a polo t-shirt"
# search_query = "looking for places to visit during the holidays"
# search_query = "An apple a day keeps the doctor away"
vec = encoder.encode(search_query)
vec.shape


# In[66]:


import numpy as np
svec = np.array(vec).reshape(1,-1)
svec.shape


# In[67]:


# faiss.normalize_L2(svec)


# ### Step 5: Search for similar vector in the FAISS index created

# In[68]:


distances, I = index.search(new_vec, k=2)
distances


# In[69]:


I


# In[70]:


I.tolist()


# In[71]:


row_indices = I.tolist()[0]
row_indices


# In[72]:


df.loc[row_indices]


# In[73]:


search_query


# You can see that the two results from the dataframe are similar to a search_query
