#!/usr/bin/env python
# coding: utf-8

# 
#                                              ( IRIS FLOWER CLASSIFICATION )

# In[1]:


#import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings(action='ignore')
import seaborn as sns


# In[14]:


#load data
data=pd.read_csv("C:/Users/jeeva/Downloads/Iris.csv")
data.head(5)


# In[3]:


#dimensions
data.ndim


# In[4]:


#null data
data.isna().sum()


# In[5]:


#duplicated data
data.duplicated().sum()


# In[6]:


#Declaring X and Y
X=data.iloc[:,1:5]
Y=data['Species']


# In[7]:


#splitting test and train data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
X_train


# In[8]:


#scatterplot using seaborn
sns.pairplot(data,hue='Species',markers=['o','s','p'])
plt.show()


# In[9]:


#histogram using matplotlib
species=data['Species']
plt.hist(species,bins=10,edgecolor='black',color=['skyblue'])
plt.xlabel('Species Variety')
plt.ylabel('Quantity')
plt.title("Quantity of Species")
plt.show()


# In[10]:


#Using KNClassifier algorithm
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
pred=knn.predict(X_test)
accuracy=accuracy_score(Y_test,pred)
print(accuracy)


# In[11]:


#Using DecisionTreeClassifier algorithm
dtc=DecisionTreeClassifier(random_state=42)
dtc.fit(X_train,Y_train)
pred=dtc.predict(X_test)
accuracy=accuracy_score(Y_test,pred)
print(accuracy)


# In[12]:


#Created new array to predict
new=np.array([[2.1,4.5,3.5,1.7]])
pred=dtc.predict(new)
print(pred)


# In[13]:


#Just rechecked my model with already values that are present
old=np.array([[6.7,3.0,5.2,2.3]])
pred=dtc.predict(old)
print(pred)

