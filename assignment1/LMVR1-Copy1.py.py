
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split


# In[3]:


RD = pd.read_csv('/home/nikhila/assignment1/winequality-white.csv',sep = ';')


# In[5]:


Y = RD.quality


# In[6]:


X = RD.drop(['quality'],axis = 1)


# In[20]:


for i in X.columns:
    X[i] = (X[i]- X[i].mean())/X[i].std()


# In[21]:


X


# In[56]:


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)


# In[57]:


test_size = 0.2


# In[58]:


random_state = 42


# In[59]:


X_train


# In[60]:


Y_train


# In[61]:


X_train['zfeature'] = np.ones(X_train.shape[0])


# In[62]:


X_train


# In[63]:


X_train.shape


# In[64]:


Y_train.shape          #not a matrix, to be converted


# In[65]:


Y_train.shape


# In[66]:


theta = np.zeros(X_train.shape[1])


# In[67]:


theta


# In[125]:


alpha = 0.3


# In[69]:


m = X_train.shape[0]              #Number of training samples


# In[70]:


n = X_train.shape[1]             #Number of features + 1 (n+1)


# In[53]:


#(n+1,1) matrix and X_train is (m,n+1) and Y_train is (m,1)


# In[37]:


h_theta = np.zeros((m,1))


# In[71]:


iters = 1000


# In[128]:


cost = np.zeros(iters)


# In[126]:


"""for i in range(10):
    pd = (alpha/m) * (X_train.T).dot(X_train.dot(theta)-Y_train)
    cost = (1/(2*m))*(((X_train.dot(theta)-Y_train).T).dot(X_train.dot(theta)-Y_train))
    theta=theta-pd
    print(cost)"""


# In[129]:


for i in range(iters):
    pd = (alpha/m) * (X_train.T).dot(X_train.dot(theta)-Y_train)
    cost[i] = (1/(2*m))*(((X_train.dot(theta)-Y_train).T).dot(X_train.dot(theta)-Y_train))
    theta=theta-pd
    
    


# In[130]:


plt.plot(cost)


# In[131]:


theta


# In[132]:


X_test['zfeature'] = np.ones(X_test.shape[0])


# In[133]:


X_test.shape


# In[134]:


theta.shape


# In[135]:


predict = (X_test).dot(theta)


# In[136]:


LSE = (1/(2*m))*((predict - Y_test).T).dot(predict - Y_test)


# In[137]:


LSE

