#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


# ## Custom Scaler

# In[72]:


# so you can imagine that the Custom Scaler is build on it



# create the Custom Scaler class

class CustomScaler(BaseEstimator,TransformerMixin): 
    
    # init or what information we need to declare a CustomScaler object
    # and what is calculated/declared as we do
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        
        # scaler is nothing but a Standard Scaler object
        self.scaler = StandardScaler(copy,with_mean,with_std)
        # with some columns 'twist'
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    
    # the fit method, which, again based on StandardScale
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    # the transform method which does the actual scaling

    def transform(self, X, y=None, copy=None):
        
        # record the initial order of the columns
        init_col_order = X.columns
        
        # scale all features that you chose when creating the instance of the class
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        
        # declare a variable containing all information that was not scaled
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        
        # return a data frame which contains all scaled features and all 'not scaled' features
        # use the original order (that you recorded in the beginning)
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# ## Abesntial Main Module

# In[151]:


class absentia_model():
    def __init__(self):
        with open('model.pkl','rb') as model,open('scaler.pkl','rb') as scalar:
            self.model = pickle.load(model)
            self.scalar = pickle.load(scalar)
        self.data = None
    
    
    def load_and_clean_data(self,datafile):
        
        df = pd.read_csv(datafile)
        data = df.copy()
        data = data.drop(['ID'],axis =1 )
        
        
        
        reason_dummies = pd.get_dummies(data['Reason for Absence'],drop_first=True)
        reason_1 = reason_dummies.iloc[:,0:14].max(axis=1)
        reason_2 = reason_dummies.iloc[:,14:17].max(axis=1)
        reason_3 = reason_dummies.iloc[:,17:20].max(axis=1)
        reason_4 = reason_dummies.iloc[:,20:28].max(axis=1)
        data = data.drop(['Reason for Absence'],axis=1)
        
        encoded_data = pd.concat([data,reason_1,reason_2,reason_3,reason_4],axis=1)
        new_columns = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',         'Daily Work Load Average', 'Body Mass Index', 'Education',                           'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1',  'Reason_2',  'Reason_3',  'Reason_4'] 
        
        encoded_data.columns = new_columns
        rearange_data = encoded_data[['Absenteeism Time in Hours','Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets',  'Reason_1',  'Reason_2',  'Reason_3',  'Reason_4']]
        
        month = []
        day = []
        for i in range(rearange_data.shape[0]):
            day.append(rearange_data['Date'][i].split('/')[0])
            month.append(rearange_data['Date'][i].split('/')[1])
        
        rearange_data['Day'] = day
        rearange_data['Month'] = month
        rearange_data['Day'] = rearange_data['Day'].astype(int)
        rearange_data['Month'] = rearange_data['Month'].astype(int)
        
        rearange_data  = rearange_data.drop(['Date','Absenteeism Time in Hours'],axis=1)
        
        all_cols = rearange_data.columns.values
        
        cols_to_remove = ['Day','Month','Daily Work Load Average','Education']
        
        to_include_col = [ i for i in all_cols if i not in cols_to_remove]
        
        self.preprocesseddata = rearange_data[to_include_col].copy()
        
        scaled_data = self.scalar.transform(rearange_data[to_include_col])
        self.data = scaled_data
        
    def predicted_probability(self):
        
        return self.model.predict_proba(self.data)[:,1]
    
    def predicted_output_category(self):
        return self.model.predict(self.data)
    
    def predicted_output(self):
        self.preprocesseddata['Probability'] = self.model.predict_proba(self.data)[:,1]
        self.preprocesseddata['Prediction'] = self.model.predict(self.data)
        return self.preprocesseddata
        
        


# In[ ]:




