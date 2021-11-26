#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:



#import the data
loan_data=pd.read_csv('/Users/sunainasram/Library/Containers/com.microsoft.Excel/Data/Downloads/loan-train.csv')


# In[ ]:


loan_data.info()


# # Insights/Data Visualisation

# In[ ]:


import matplotlib.pyplot as plt
plt.hist(loan_data['Loan_Status'])


# In[ ]:


#pd.crosstab(loan_data.ApplicantIncome,loan_data.LoanAmount).plot(kind= "Bar")
#plt.barh('Loan_data',height=20,width=1000)
#plt.barh('Loan_data',height=20,width=1000)


# In[ ]:


import seaborn as sns


# In[ ]:




pd.crosstab(loan_data.Loan_Status,loan_data.Gender)


# In[ ]:


sns.countplot(y='Gender',hue='Loan_Status',data=loan_data)


# In[ ]:


##INSIGHT- Males have availed more loans compared to females. 
#gRADUATES HAVE AVAILED MORE  LOAN COMPARED TO NON-GRADUATES


# In[ ]:


sns.countplot(y='Education',hue='Loan_Status',data=loan_data)


# In[ ]:


sns.countplot(y='Property_Area',hue='Loan_Status',data=loan_data)


# In[ ]:


##Semiurban obtain more loan, folowed by Urban and then rural. 


# In[ ]:


sns.countplot(y='Loan_Amount_Term',hue='Loan_Status',data=loan_data)


# In[ ]:


##High number of them go for a 360 cyclic loan term.
#pay back within a year
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:



#grid=sns.FacetGrid(loan_data,row='Gender',col='Married',size=2.2,aspect=1.6,)
#grid.map(plt.hist,'ApplicantIncome',alpha=0.5,bins=10)
#grid.addLengend()


# In[ ]:



grid=sns.FacetGrid(loan_data,row='Gender',col='Education',size=2.2,aspect=1.6,)
grid.map(plt.hist,'ApplicantIncome',alpha=0.5,bins=10)
grid.addLengend()


# In[ ]:



grid=sns.FacetGrid(loan_data,row='Married',col='Education',size=2.2,aspect=1.6,)
grid.map(plt.hist,'ApplicantIncome',alpha=0.5,bins=10)
grid.addLengend()


# In[ ]:



grid=sns.FacetGrid(loan_data,row='Self_Employed',col='Education',size=2.2,aspect=1.6,)
grid.map(plt.hist,'ApplicantIncome',alpha=0.5,bins=10)
grid.addLengend()


# In[ ]:



grid=sns.FacetGrid(loan_data,row='Married',col='Dependents',size=2.2,aspect=1.6,)
grid.map(plt.hist,'ApplicantIncome',alpha=0.5,bins=10)
grid.addLengend()


# In[ ]:


grid=sns.FacetGrid(loan_data,row='Self_Employed',col='Dependents',size=2.2,aspect=1.6,)
grid.map(plt.hist,'ApplicantIncome',alpha=0.5,bins=10)
grid.addLengend()


# In[ ]:



grid=sns.FacetGrid(loan_data,row='Education',col='Dependents',size=2.2,aspect=1.6,)
grid.map(plt.hist,'ApplicantIncome',alpha=0.5,bins=10)
grid.addLengend()


# In[ ]:



grid=sns.FacetGrid(loan_data,row='Married',col='Credit_History',size=2.2,aspect=1.6,)
grid.map(plt.hist,'ApplicantIncome',alpha=0.5,bins=10)
grid.addLengend()


# In[ ]:


loan_data.isnull()


# In[ ]:


loan_data.describe()


# In[ ]:


loan_data.head()


# In[ ]:


loan_data.isnull().value_counts


# In[ ]:


loan_data.isnull().sum()


# In[ ]:


loan_data.dropna


# In[ ]:


loan_data.drop(["Loan_ID"],axis=1,inplace=True)


# In[ ]:


loan_data.head()


# In[ ]:


loan_data.fillna(0)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


from sklearn.preprocessing import LabelEncoder
import pandas as pd


# In[ ]:


loan_data=loan_data.copy()
loan_data.head()


# In[ ]:


labelencoder = LabelEncoder()


# In[ ]:


loan_data.iloc[:, 0]=labelencoder.fit_transform(loan_data.iloc[:,0])


# In[ ]:


loan_data.iloc[:, 1]=labelencoder.fit_transform(loan_data.iloc[:,1])


# In[ ]:


loan_data.iloc[:, 3]=labelencoder.fit_transform(loan_data.iloc[:,3])


# In[ ]:


loan_data.iloc[:, 4]=labelencoder.fit_transform(loan_data.iloc[:,4])


# In[ ]:


loan_data.iloc[:, -1]=labelencoder.fit_transform(loan_data.iloc[:,-1])


# In[ ]:


loan_data.iloc[:, -2]=labelencoder.fit_transform(loan_data.iloc[:,-2])


# In[ ]:


loan_data.head()


# In[ ]:


loan_data.isnull().sum()
#loan_data.info()


# In[ ]:


#Mean Imputation
mean = loan_data['LoanAmount'].mean()
print(mean)

loan_data['LoanAmount'] = loan_data['LoanAmount'].fillna(mean)


# In[ ]:


mean = loan_data['Loan_Amount_Term'].mean()
print(mean)

loan_data['Loan_Amount_Term'] = loan_data['Loan_Amount_Term'].fillna(mean)


# In[ ]:


mean = loan_data['LoanAmount'].mean()
print(mean)

loan_data['LoanAmount'] = loan_data['LoanAmount'].fillna(mean)


# In[ ]:


mean = loan_data['Credit_History'].mean()
print(mean)

loan_data['Credit_History'] = loan_data['Credit_History'].fillna(mean)


# In[ ]:


loan_data.head()


# In[ ]:


loan_data.isnull().sum()


# In[ ]:


#pd.Series.astype
#loan_data["Loan_Status"] = loan_data.Loan_Status.astype(float)


# In[ ]:


loan_data.drop(['Dependents'],axis=1,inplace=True)


# In[ ]:


loan_data.head().isnull().sum()
loan_data.head()


# In[ ]:


plt.hist(loan_data['Loan_Status'])
plt.show()
plt.scatter(loan_data.ApplicantIncome,loan_data.LoanAmount)


# In[ ]:


x=loan_data.iloc[:,0:10]
y=loan_data.iloc[:,-1]


# In[ ]:


x


# In[ ]:


y


# In[ ]:


x.info()


# In[ ]:


y.isnull().value_counts


# In[ ]:


loan_data.isnull().value_counts()
loan_data.isnull().sum()
loan_data.isna().sum()


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=10)


# In[ ]:


#sns.countplot(y='Married',hue='LoanAmount',data=x_train)


# # Data viz with insights

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:



grid=sns.FacetGrid(loan_data,row='Gender',col='Married',size=2.2,aspect=1.6,)
grid.map(plt.hist,'ApplicantIncome',alpha=0.5,bins=10)
grid.addLengend()


# # Logistic Regression Model

# In[ ]:


#Logistic regression and fit the model
classifier = LogisticRegression(max_iter=100)
classifier.fit(x_train,y_train)


# In[ ]:


#Predict for X dataset
y_pred = classifier.predict(x_test)


# In[ ]:


y_pred_df= pd.DataFrame({'actual': y_test,
                         'predicted_prob': classifier.predict(x_test)})


# In[ ]:


y_pred_df
classifier.score(x_test,y_test)


# In[ ]:


# Confusion Matrix for the model accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_pred)
print (confusion_matrix)


# In[ ]:


#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# # Decision Tree Model

# In[ ]:


model1 = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model1.fit(x_train,y_train)


# In[ ]:


model1.get_n_leaves()


# In[ ]:


preds = model1.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[ ]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[ ]:


print(classification_report(preds,y_test))


# # Bagged decision tree Model

# In[ ]:


# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
#filename = '/Users/sunainasram/Library/Containers/com.microsoft.Excel/Data/Downloads/pima-indians-diabetes.data.csv-2.xls'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
x=loan_data.iloc[:,0:10]
y=loan_data.iloc[:,-1]

seed = 7

kfold = KFold(n_splits=10)
cart = DecisionTreeClassifier()
num_trees = 100
model2 = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model2, x, y, cv=kfold)
print(results.mean())


# # Random Forest Model

# In[ ]:


# Random Forest Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

#X = array[:,0:8]
#Y = array[:,8]

x=loan_data.iloc[:,0:10]
y=loan_data.iloc[:,-1]

num_trees = 100
max_features = 3
kfold = KFold(n_splits=10)
model3 = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model3, x, y, cv=kfold)
print(results.mean())


# # ADA Boost Model

# In[ ]:


# AdaBoost Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
#filename = '/Users/sunainasram/Library/Containers/com.microsoft.Excel/Data/Downloads/pima-indians-diabetes.data.csv-2.xls'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values

x=loan_data.iloc[:,0:10]
y=loan_data.iloc[:,-1]

num_trees = 10
seed=7
kfold = KFold(n_splits=10)
model4 = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model4, x, y, cv=kfold)
print(results.mean())
from sklearn import metrics


# # Stacking Model

# In[ ]:


# Stacking Ensemble for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
#filename = '/Users/sunainasram/Library/Containers/com.microsoft.Excel/Data/Downloads/pima-indians-diabetes.data.csv-2.xls'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]

x=loan_data.iloc[:,0:10]
y=loan_data.iloc[:,-1]

kfold = KFold(n_splits=10)

# create the sub models
estimators = []
model5 = LogisticRegression(max_iter=500)
estimators.append(('logistic', model5))
model6 = DecisionTreeClassifier()
estimators.append(('cart', model6))
model7 = SVC()
estimators.append(('svm', model7))

# create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, x, y, cv=kfold)
print(results.mean())


# In[ ]:


model5.fit(x_train,y_train)
model5.predict(x_test)
model5.score(x_test,y_test)


# In[ ]:


model6.fit(x_train,y_train)
model6.predict(x_test)


model6.score(x_test,y_test)


# In[ ]:


model7.fit(x_train,y_train)
model7.predict(x_test)


model7.score(x_test,y_test)


# # ADA Boost perfomed better than Logistic regression, DT, RF,Bagged DT. 
# ## Accuracy =80.30%

# In[ ]:





# In[ ]:





# In[ ]:


get_ipython().run_line_magic('pip', 'freeze > requirements.txt')


# In[1]:


cat requirements.txt


# In[ ]:




