# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 20:59:50 2018

@author: manish
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns


data=pd.read_csv('train.csv')

print(data.head())

y=data['Survived']

cols_to_remove=['PassengerId','Survived','Name','Ticket','Cabin']
data1 = data.drop(cols_to_remove,axis=1)

l1=LabelEncoder()
data1['sex']=l1.fit_transform(data['Sex'])

l2=LabelEncoder()
data1['embarked']=l2.fit_transform(data['Embarked'].astype(str))

data1 = data1.drop(['Sex','Embarked'],axis=1)

cater_cols=['Pclass','SibSp','Parch','sex','embarked']
contin_cols=['Age','Fare']
total_df=cater_cols+contin_cols

conti=pd.DataFrame(data1,columns=contin_cols)
cater=pd.DataFrame(data1,columns=cater_cols)

impca=Imputer(missing_values='NaN',strategy='most_frequent')
impca_out=impca.fit_transform(cater)

impca_df=pd.DataFrame(impca_out,columns=cater_cols)


impco=Imputer(missing_values='NaN',strategy='mean')
impo_out = impco.fit_transform(conti)
impco_df=pd.DataFrame(impo_out,columns=contin_cols)

total_df_data=pd.concat([impca_df,impco_df],axis=1)

sl1=StandardScaler()
sl2=sl1.fit_transform(total_df_data)
sl3=pd.DataFrame(sl2,columns=total_df)

X_train,X_test,Y_train,Y_test=train_test_split(sl3,y,test_size=0.2,random_state=42)

rfe=DecisionTreeClassifier()
rfe.fit(X_train,Y_train)

y_pred=rfe.predict(X_test)

print(accuracy_score(y_pred,Y_test))

sns.heatmap(data=total_df_data.corr(),annot=True)
for i in data1:
    for j in data1:
        if(i==j):
            print(data1[i][j],'same column')
        else:
            print(data[i][j],'different ')
            
            
a=set()
for i in data1:
    for j in data1:
        if(i==j):
            continue
        else:
            if(data1[i][j]>0.1):
                a.add(i)
print(a)
            
            
            
            
            
            
            # -*- coding: utf-8 -*-



