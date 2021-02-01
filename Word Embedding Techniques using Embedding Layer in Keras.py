#!/usr/bin/env python
# coding: utf-8

# ## Word Embedding Techniques using Embedding Layer in Keras

# ### Import Important Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
import re
import nltk


# In[2]:


import keras
import tensorflow


# In[3]:


from keras.preprocessing.text import one_hot


# In[4]:


### sentences
sent=['the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good']


# In[5]:


print(sent)


# In[6]:


sent


# In[7]:


# Vocabulary size
voc_size=10000


# ### One Hot Representation

# In[8]:


onehot_rep=[one_hot(words,voc_size) for words in sent]


# In[9]:


print(onehot_rep)


# ### Word Embedding Represntation

# In[10]:


from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences


# In[11]:


sent_length=8


# In[12]:


embedded_docs=pad_sequences(onehot_rep,
    maxlen=sent_length,
    padding='pre')


# In[13]:


print(embedded_docs)


# In[14]:


dim=10


# In[15]:


model=Sequential()


# In[16]:


model.add(Embedding(voc_size,10,input_length=sent_length))


# In[17]:


model.compile(optimizer='Adam',loss='mse',metrics=['accuracy'])


# In[18]:


model.summary()


# In[19]:


print(model.predict(embedded_docs))


# In[20]:


embedded_docs[0]


# In[21]:


model.predict(embedded_docs[0])


# In[ ]:




