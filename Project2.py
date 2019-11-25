#!/usr/bin/env python
# coding: utf-8

# # PROJECT2
# SALIFOU SYLLA
# 
# NOVEMBER 18, 2019

# # Multiple Linear Regression: Wine quality prediction
# 
# The dataset used is Wine Quality Data set from University of California Irvine Machine Learning Repository. It is related to red variants of the Portuguese “Vinho Verde” wine. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.). We will take into account various input features like fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol. Based on these features we will predict the quality of the wine.

# In[118]:


import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn import metrics 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[123]:


dataset = pd.read_csv('RedWine.csv')
print(dataset)
type(dataset)


# In[127]:


dataset.describe()


# In[100]:


dataset.dtypes


# ## Correlations between each attribute of dataset 

# In[101]:


# Given the set of values for features, we have to predict the quality of wine. 
# finding correlation of each feature with our target variable - quality
correlations = dataset.corr()['quality'].drop('quality')
print(correlations)


# ## Heatmap to get a detailed diagram of correlation 

# In[102]:


sns.heatmap(dataset.corr())
plt.show()


# # Input features (X) - Output target (Y)
# Create vectors x containing input features and y containing the quality variable. In x we get all the features except residual sugar.Outputs only those features whose correlation is above a threshold value (0.05).

# In[103]:


X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values
y = dataset['quality'].values


# In[104]:


#Let's check the average value of the “quality” column.
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['quality'])


# In[105]:


#Next, we split 80% of the data to the training set while 20% of the data to test set using below code.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[106]:


#Now lets train our model.
regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# # Coefficients our regression model has chosen

# In[107]:


#In the case of multivariable linear regression, the regression model has to find the most optimal coefficients for all the attributes
#coeffecients = pd.DataFrame(regressor.coef_, X.columns) 
#coeffecients.columns = ['Coeffecient'] 
#print(coeffecients)
print(regressor.coef_)


# These numbers mean that holding all other features fixed, a 1 unit increase in sulphates will lead to an increase of 0.88 in quality of wine, and similarly for the other features.
# Also holding all other features fixed, a 1 unit increase in volatile acidity will lead to a decrease of 1.155 in quality of wine, and similarly for the other features.

# # Now let's do prediction on test data.

# In[109]:


y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.head(25)


# #  Histogram of the residuals
# 

# In[110]:


#Now let's check the histogram of the residuals. Does it satisfy the assumptions for inference?
plt1 = plt.hist(y_test - y_pred)


# # Scatterplot of predicted values and residuals 

# In[113]:


plt.scatter(y_pred, y_test - y_pred)


# # Now let's plot the comparison of Actual and Predicted values

# In[116]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.head(15).plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# ### This model has returned pretty good prediction results.

# #  Performance of the algorithm
#  

# In[117]:


# We’ll do this by finding the values for MAE, MSE, and RMSE
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# The value of root mean squared error is 0.62, which is slightly greater than 10% of the mean value which is 5.63. This means that our algorithm was not very accurate but can still make reasonably good predictions.

# # Factors contributing to the inaccuracy

# Data size: We need to have a huge amount of data to get the best possible prediction.
# Bad assumptions: We made the assumption that this data has a linear relationship, but that might not be the case. Visualizing the data may help you determine that.
# Choice of features:Features used should have a high correlation to the values we were trying to predict.
